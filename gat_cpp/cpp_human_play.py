"""
终端人机对战入口（支持 n*n 棋盘）

设计目标：
1) 先提供可用的 CLI 版本，方便直接用已有 8x8 模型测试。
2) 业务逻辑与终端 IO 解耦，为后续 UI 版本预留可复用接口。
"""

import argparse
import os
import pickle
import shutil
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
	sys.path.insert(0, ROOT_DIR)

try:
	from env.game import Board
	from env.mcts_alphaZero import MCTSPlayer
except ModuleNotFoundError:
	# 兼容当前仓库结构：脚本与 cpp_game/cpp_mcts_alphaZero 位于同一目录 (gat_cpp)
	from cpp_game import Board
	from cpp_mcts_alphaZero import MCTSPlayer
from pipeline.policy_value_net import PolicyValueNet


def _setup_windows_dll_dirs(extra_dirs: List[str]) -> None:
	if os.name != "nt" or not hasattr(os, "add_dll_directory"):
		return

	runtime_candidates = [
		os.path.dirname(shutil.which("c++.exe") or ""),
		os.path.dirname(shutil.which("g++.exe") or ""),
		"C:/msys64/ucrt64/bin",
		"C:/msys64/mingw64/bin",
	]

	dll_dirs: List[str] = []
	for p in list(extra_dirs) + runtime_candidates:
		if p and os.path.isdir(p) and p not in dll_dirs:
			dll_dirs.append(p)

	for p in dll_dirs:
		os.add_dll_directory(p)


def _import_mcts_cpp_module():
	build_release = os.path.join(ROOT_DIR, "gat_cpp", "build", "Release")
	build_root = os.path.join(ROOT_DIR, "gat_cpp", "build")
	_setup_windows_dll_dirs([build_release, build_root])
	for p in [build_release, build_root]:
		if p not in sys.path:
			sys.path.insert(0, p)

	import importlib
	return importlib.import_module("mcts_cpp")


@dataclass
class StepResult:
	"""一次落子后的结果，方便 UI 层直接消费。"""

	move: int
	location: Tuple[int, int]
	player: int
	is_end: bool
	winner: int


class ModelPolicyAdapter:
	"""把自定义网络包装成 MCTS 可调用的 policy_value_fn。"""

	def __init__(self, board_size: int, device: str = "cpu"):
		self.board_size = board_size
		self.device = device
		self.net = PolicyValueNet(board_size=board_size, num_channels=64, device=device)
		self.legacy_params = None

	def _to_tensor(self, arr):
		if torch.is_tensor(arr):
			return arr.detach().to(self.device, dtype=torch.float32)
		return torch.tensor(np.asarray(arr), dtype=torch.float32, device=self.device)

	def _activate_legacy_params(self, raw_params) -> None:
		if len(raw_params) != 16:
			raise ValueError(f"legacy 模型参数数量错误: 期望 16，实际 {len(raw_params)}")
		if self.board_size != 8:
			raise ValueError("legacy .model/.model2 仅支持 8x8 棋盘")

		params = [self._to_tensor(x) for x in raw_params]
		self.legacy_params = params

	def _legacy_forward(self, state_tensor: torch.Tensor):
		w1, b1, w2, b2, w3, b3, wp_conv, bp_conv, wp_fc, bp_fc, wv_conv, bv_conv, wv_fc1, bv_fc1, wv_fc2, bv_fc2 = self.legacy_params

		x = F.relu(F.conv2d(state_tensor, w1, b1, padding=1))
		x = F.relu(F.conv2d(x, w2, b2, padding=1))
		x = F.relu(F.conv2d(x, w3, b3, padding=1))

		p = F.relu(F.conv2d(x, wp_conv, bp_conv, padding=0))
		p = p.reshape(p.shape[0], -1)
		if wp_fc.shape[0] == p.shape[1]:
			policy_logits = p @ wp_fc + bp_fc
		elif wp_fc.shape[1] == p.shape[1]:
			policy_logits = p @ wp_fc.T + bp_fc
		else:
			raise ValueError(f"legacy policy fc 维度不匹配: input={p.shape}, w={wp_fc.shape}")
		policy = torch.softmax(policy_logits, dim=-1)

		v = F.relu(F.conv2d(x, wv_conv, bv_conv, padding=0))
		v = v.reshape(v.shape[0], -1)
		if wv_fc1.shape[0] == v.shape[1]:
			v = F.relu(v @ wv_fc1 + bv_fc1)
		elif wv_fc1.shape[1] == v.shape[1]:
			v = F.relu(v @ wv_fc1.T + bv_fc1)
		else:
			raise ValueError(f"legacy value fc1 维度不匹配: input={v.shape}, w={wv_fc1.shape}")

		if wv_fc2.shape[0] == v.shape[1]:
			value = torch.tanh(v @ wv_fc2 + bv_fc2)
		elif wv_fc2.shape[1] == v.shape[1]:
			value = torch.tanh(v @ wv_fc2.T + bv_fc2)
		else:
			raise ValueError(f"legacy value fc2 维度不匹配: input={v.shape}, w={wv_fc2.shape}")

		return policy, value

	def _load_legacy_pickle_file(self, model_path: str) -> bool:
		if not model_path.lower().endswith((".model", ".model2")):
			return False
		with open(model_path, "rb") as f:
			legacy_obj = pickle.load(f, encoding="latin1")
		if isinstance(legacy_obj, (list, tuple)):
			self._activate_legacy_params(list(legacy_obj))
			return True
		raise ValueError("legacy 模型格式错误：期望 list/tuple")

	def load_checkpoint(self, model_path: str) -> None:
		if not os.path.exists(model_path):
			raise FileNotFoundError(f"模型文件不存在: {model_path}")

		if self._load_legacy_pickle_file(model_path):
			return

		try:
			checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
		except TypeError:
			# 兼容旧版 PyTorch（无 weights_only 参数）
			checkpoint = torch.load(model_path, map_location="cpu")

		if isinstance(checkpoint, dict) and "legacy_params" in checkpoint:
			self._activate_legacy_params(checkpoint["legacy_params"])
			return
		if isinstance(checkpoint, (list, tuple)):
			self._activate_legacy_params(list(checkpoint))
			return

		params, _ = self.net.get_all_params()

		if isinstance(checkpoint, dict) and "model_state" in checkpoint:
			model_state = checkpoint["model_state"]
		elif isinstance(checkpoint, dict):
			# 兼容直接保存参数字典的情况
			model_state = checkpoint
		else:
			raise ValueError("不支持的模型文件格式，预期为 dict/checkpoint")

		loaded_keys = 0
		for name, tensor in model_state.items():
			if name in params and params[name] is not None and torch.is_tensor(tensor):
				if tuple(params[name].shape) == tuple(tensor.shape):
					params[name].data.copy_(tensor.to(params[name].device))
					loaded_keys += 1

		if loaded_keys == 0:
			raise ValueError("模型加载失败：未匹配到任何网络参数，请检查棋盘尺寸或权重文件")

	def policy_value_fn(self, board: Board):
		legal_positions = board.availables
		current_state = np.ascontiguousarray(
			board.current_state().reshape(-1, 4, self.board_size, self.board_size)
		)
		state_tensor = torch.tensor(current_state, dtype=torch.float32, device=self.device)

		with torch.no_grad():
			if self.legacy_params is not None:
				act_probs_tensor, value_tensor = self._legacy_forward(state_tensor)
			else:
				self.net.eval_mode()
				act_probs_tensor, value_tensor = self.net.forward(state_tensor)

		act_probs_full = act_probs_tensor.cpu().numpy()[0]
		board_moves = self.board_size * self.board_size
		if act_probs_full.shape[0] == board_moves + 1:
			# 新版网络包含 pass 动作
			act_probs = act_probs_full[:-1]
		elif act_probs_full.shape[0] == board_moves:
			# legacy 网络不包含 pass 动作
			act_probs = act_probs_full
		else:
			raise ValueError(f"策略头输出维度异常: {act_probs_full.shape[0]}")

		value = float(value_tensor.cpu().numpy()[0][0])

		if not legal_positions:
			return [], value

		legal_probs = act_probs[legal_positions]
		prob_sum = np.sum(legal_probs)
		if prob_sum > 0:
			legal_probs = legal_probs / prob_sum
		else:
			legal_probs = np.full(len(legal_positions), 1.0 / len(legal_positions), dtype=np.float32)

		return zip(legal_positions, legal_probs), value


class HumanAIGameSession:
	"""
	对局会话（UI 无关）：
	- 管理棋盘与 MCTS 玩家
	- 提供“人类落子 / AI 落子 / 状态快照”接口
	"""

	def __init__(
		self,
		board_size: int,
		n_in_row: int,
		model_path: Optional[str],
		c_puct: float = 5.0,
		n_playout: int = 120,
		human_first: bool = True,
		device: Optional[str] = None,
		engine: str = "python",
		cpp_threads: int = 0,
	):
		if board_size < n_in_row:
			raise ValueError("board_size 不能小于 n_in_row")
		self.board_size = board_size
		self.n_in_row = n_in_row
		self.model_path = model_path
		self.engine = engine
		self.cpp_threads = cpp_threads
		self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

		self.board = Board(width=board_size, height=board_size, n_in_row=n_in_row)
		self.ai_player = None
		self.policy_adapter = None
		self.cpp_module = None
		self.cpp_board = None
		self.cpp_ai = None

		if self.engine == "python":
			if not model_path:
				raise ValueError("python 引擎需要提供模型路径")
			self.policy_adapter = ModelPolicyAdapter(board_size=board_size, device=self.device)
			self.policy_adapter.load_checkpoint(model_path)

			self.ai_player = MCTSPlayer(
				policy_value_function=self.policy_adapter.policy_value_fn,
				c_puct=c_puct,
				n_playout=n_playout,
				is_selfplay=0,
			)
		elif self.engine == "cpp":
			self.cpp_module = _import_mcts_cpp_module()
			self.cpp_board = self.cpp_module.Board(board_size, board_size, n_in_row)
			self.cpp_ai = self.cpp_module.AlphaZeroPlayer(
				c_puct=int(c_puct),
				n_playout=int(n_playout),
				num_threads=int(cpp_threads),
			)
		else:
			raise ValueError(f"不支持的 engine: {self.engine}")

		# 约定：玩家编号固定为 1(人类) 和 2(AI)
		self.human_player_id = 1
		self.ai_player_id = 2
		self.reset(human_first=human_first)

	def reset(self, human_first: bool = True) -> None:
		start_player = 0 if human_first else 1
		self.board.init_board(start_player=start_player)
		if self.engine == "python":
			self.ai_player.set_player_ind(self.ai_player_id)
			self.ai_player.reset_player()
		else:
			self.cpp_board.init_board(start_player=start_player)
			self.cpp_ai.set_player_ind(self.ai_player_id)
			self.cpp_ai.reset_player()

	def is_human_turn(self) -> bool:
		return self.board.current_player == self.human_player_id

	def parse_human_input(self, raw_text: str) -> int:
		text = raw_text.strip()
		if not text:
			return -1

		text = text.replace("，", ",")
		if "," in text:
			parts = [p.strip() for p in text.split(",")]
		else:
			parts = [p.strip() for p in text.split()]

		if len(parts) != 2:
			return -1

		try:
			row, col = int(parts[0]), int(parts[1])
		except ValueError:
			return -1

		return self.board.location_to_move([row, col])

	def _apply_move(self, move: int, player: int) -> StepResult:
		self.board.do_move(move)
		is_end, winner = self.board.game_end()
		row, col = self.board.move_to_location(move)
		return StepResult(
			move=move,
			location=(row, col),
			player=player,
			is_end=is_end,
			winner=winner,
		)

	def apply_human_move(self, move: int) -> StepResult:
		if self.board.current_player != self.human_player_id:
			raise RuntimeError("当前不是人类回合")
		if move not in self.board.availables:
			raise ValueError("非法落子")

		if self.engine == "cpp":
			self.cpp_board.do_move(move)

		result = self._apply_move(move, self.human_player_id)
		# 同步 AI 搜索树到最新局面
		if self.engine == "python":
			self.ai_player.update_with_move(move)
		else:
			self.cpp_ai.update_with_move(move)
		return result

	def apply_ai_move(self, temp: float = 1e-3) -> StepResult:
		if self.board.current_player != self.ai_player_id:
			raise RuntimeError("当前不是 AI 回合")

		if self.engine == "python":
			move = self.ai_player.get_action(self.board, temp=temp, return_prob=0)
		else:
			move = self.cpp_ai.get_action(self.cpp_board)
		if move == -1:
			is_end, winner = self.board.game_end()
			return StepResult(move=-1, location=(-1, -1), player=self.ai_player_id, is_end=is_end, winner=winner)

		if self.engine == "cpp":
			self.cpp_board.do_move(move)
			self.cpp_ai.update_with_move(move)

		return self._apply_move(move, self.ai_player_id)

	def get_snapshot(self) -> Dict[str, object]:
		"""返回给 UI 的标准状态快照。"""
		grid: List[List[int]] = [[0 for _ in range(self.board_size)] for _ in range(self.board_size)]
		for move, player in self.board.states.items():
			r, c = self.board.move_to_location(move)
			grid[r][c] = player

		is_end, winner = self.board.game_end()
		return {
			"board_size": self.board_size,
			"n_in_row": self.n_in_row,
			"current_player": self.board.current_player,
			"human_player": self.human_player_id,
			"ai_player": self.ai_player_id,
			"last_move": self.board.last_move,
			"is_end": is_end,
			"winner": winner,
			"availables": list(self.board.availables),
			"grid": grid,
		}


class TerminalRenderer:
	"""终端渲染层，后续替换成 UI 层时不影响对局逻辑。"""

	@staticmethod
	def render(session: HumanAIGameSession) -> None:
		board = session.board
		width = board.width
		print()
		print(f"棋盘: {width}x{width}, 连珠: {board.n_in_row}")
		print(f"你是玩家 {session.human_player_id}(X), AI 是玩家 {session.ai_player_id}(O)")

		print("   " + " ".join(f"{x:2d}" for x in range(width)))
		for h in range(width):
			row_cells = []
			for w in range(width):
				loc = h * width + w
				p = board.states.get(loc, 0)
				if p == session.human_player_id:
					cell = "X"
				elif p == session.ai_player_id:
					cell = "O"
				else:
					cell = "."
				row_cells.append(f"{cell:2s}")
			print(f"{h:2d} " + " ".join(row_cells))
		print()


def _find_default_model_path() -> str:
	candidates = [
		os.path.join(ROOT_DIR, "models", "healthy_policy.pth"),
		os.path.join(ROOT_DIR, "models", "current_policy.pth"),
		os.path.join(ROOT_DIR, "env", "current_policy.pth"),
	]
	for path in candidates:
		if os.path.exists(path):
			return path
	# 如果都不存在，返回首选路径，便于提示信息更直观
	return candidates[0]


def run_terminal_game(args: argparse.Namespace) -> None:
	model_path = args.model if args.model else _find_default_model_path()
	if args.engine == "cpp" and not args.model:
		model_path = None
	session = HumanAIGameSession(
		board_size=args.size,
		n_in_row=args.n_in_row,
		model_path=model_path,
		c_puct=args.c_puct,
		n_playout=args.n_playout,
		human_first=(not args.ai_first),
		device=args.device,
		engine=args.engine,
		cpp_threads=args.cpp_threads,
	)

	print("=" * 64)
	print("Gomoku Terminal Human vs AI")
	if args.engine == "python":
		print(f"engine: python, model: {model_path}")
	else:
		print(f"engine: cpp, threads: {args.cpp_threads}")
	print("输入格式: row,col 或 row col，例如: 3,4")
	print("命令: q/quit/exit 退出, h/help 查看帮助")
	print("坐标从 0 开始")
	print("=" * 64)

	while True:
		TerminalRenderer.render(session)
		is_end, winner = session.board.game_end()
		if is_end:
			if winner == -1:
				print("游戏结束：平局")
			elif winner == session.human_player_id:
				print("游戏结束：你赢了")
			else:
				print("游戏结束：AI 获胜")
			return

		if session.is_human_turn():
			raw = input(f"你的回合(玩家 {session.human_player_id}) > ").strip()
			cmd = raw.lower()
			if cmd in {"q", "quit", "exit"}:
				print("已退出对局")
				return
			if cmd in {"h", "help"}:
				print("帮助: 输入 row,col 或 row col 进行落子，例: 2,5")
				continue

			move = session.parse_human_input(raw)
			if move == -1 or move not in session.board.availables:
				print("输入无效或该位置不可落子，请重试")
				continue

			result = session.apply_human_move(move)
			print(f"你落子: ({result.location[0]}, {result.location[1]})")
		else:
			print("AI 思考中...")
			result = session.apply_ai_move(temp=args.ai_temp)
			if result.move == -1:
				print("AI 无合法动作")
			else:
				print(f"AI 落子: ({result.location[0]}, {result.location[1]})")


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="n*n Gomoku 终端人机对战")
	parser.add_argument("--size", type=int, default=8, help="棋盘边长 N（n*n）")
	parser.add_argument("--n-in-row", type=int, default=5, help="连珠数")
	parser.add_argument("--model", type=str, default=None, help="模型路径 (.pth)，默认自动探测")
	parser.add_argument("--n-playout", type=int, default=120, help="每步 MCTS 模拟次数")
	parser.add_argument("--c-puct", type=float, default=5.0, help="PUCT 探索系数")
	parser.add_argument("--ai-temp", type=float, default=1e-3, help="AI 落子温度")
	parser.add_argument("--engine", type=str, default="python", choices=["python", "cpp"], help="AI 引擎后端")
	parser.add_argument("--cpp-threads", type=int, default=0, help="C++ 引擎线程数，0 表示自动")
	parser.add_argument("--ai-first", action="store_true", help="AI 先手")
	parser.add_argument(
		"--device",
		type=str,
		default=None,
		choices=["cpu", "cuda"],
		help="推理设备，默认自动选择",
	)
	return parser


if __name__ == "__main__":
	cli_args = build_arg_parser().parse_args()
	run_terminal_game(cli_args)
