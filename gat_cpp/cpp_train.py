import argparse
import csv
import json
import logging
import multiprocessing
import os
import queue
import random
import shutil
import sys
import threading
import time
from collections import defaultdict, deque
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from cpp_game import Board, Game
from cpp_mcts_alphaZero import MCTSPlayer
from cpp_mcts_pure import MCTSPlayer as MCTS_Pure

# 导入手搓组件
from pipeline.policy_value_net import PolicyValueNet
from pipeline.optimizer import Adam
from pipeline.losses import combined_loss


_MCTS_CPP_MODULE = None


def _setup_windows_dll_dirs(extra_dirs):
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return

    runtime_candidates = [
        os.path.dirname(shutil.which("c++.exe") or ""),
        os.path.dirname(shutil.which("g++.exe") or ""),
        "C:/msys64/ucrt64/bin",
        "C:/msys64/mingw64/bin",
    ]

    dll_dirs = []
    for p in list(extra_dirs) + runtime_candidates:
        if p and os.path.isdir(p) and p not in dll_dirs:
            dll_dirs.append(p)

    for p in dll_dirs:
        os.add_dll_directory(p)


def _import_mcts_cpp_module():
    global _MCTS_CPP_MODULE
    if _MCTS_CPP_MODULE is not None:
        return _MCTS_CPP_MODULE

    build_release = os.path.join(ROOT_DIR, "gat_cpp", "build", "Release")
    build_root = os.path.join(ROOT_DIR, "gat_cpp", "build")
    _setup_windows_dll_dirs([build_release, build_root])
    for p in [build_release, build_root]:
        if p not in sys.path:
            sys.path.insert(0, p)

    import importlib

    _MCTS_CPP_MODULE = importlib.import_module("mcts_cpp")
    return _MCTS_CPP_MODULE


class CppSelfPlayPlayer:
    """C++ 并行 MCTS 自对弈玩家，产出与 Python MCTSPlayer 同构的数据。"""

    def __init__(self, board_size, n_in_row, c_puct=5, n_playout=400, num_threads=0,
                 is_selfplay=True, seed=None, eval_callback=None, eval_batch_size=8):
        self.board_size = board_size
        self.n_in_row = n_in_row
        self.c_puct = c_puct
        self.n_playout = n_playout
        self.num_threads = num_threads
        self.is_selfplay = is_selfplay
        self.player = None
        self.seed = int(seed) if seed is not None else None
        self.rng = np.random.default_rng(self.seed)

        self.mcts_cpp = _import_mcts_cpp_module()
        self.c_board = self.mcts_cpp.Board(board_size, board_size, n_in_row)
        self.ai = self.mcts_cpp.AlphaZeroPlayer(
            c_puct=int(c_puct),
            n_playout=int(n_playout),
            seed=(int(seed) if seed is not None else 42),
            num_threads=int(num_threads),
        )
        # Inject neural network callback for NN-guided MCTS
        if eval_callback is not None:
            self.ai.set_eval_callback(eval_callback, int(eval_batch_size))

    def set_player_ind(self, p):
        self.player = p
        self.ai.set_player_ind(p)

    def reset_player(self):
        self.c_board.init_board(start_player=0)
        self.ai.reset_player()

    def update_with_move(self, move):
        self.c_board.do_move(move)
        self.ai.update_with_move(move)

    def get_action(self, board, temp=1.0, return_prob=1):
        probs_full = np.asarray(self.ai.get_move_probs(self.c_board, float(temp)), dtype=np.float32)
        legal_moves = list(board.availables)
        if not legal_moves:
            if return_prob:
                return -1, np.zeros(self.board_size * self.board_size, dtype=np.float32)
            return -1

        legal_probs = probs_full[legal_moves]
        prob_sum = float(np.sum(legal_probs))
        if prob_sum > 0:
            legal_probs = legal_probs / prob_sum
        else:
            legal_probs = np.full(len(legal_moves), 1.0 / len(legal_moves), dtype=np.float32)

        move_probs = np.zeros(self.board_size * self.board_size, dtype=np.float32)
        move_probs[legal_moves] = legal_probs

        acts = np.asarray(legal_moves, dtype=np.int64)
        if self.is_selfplay:
            noisy_probs = 0.75 * legal_probs + 0.25 * self.rng.dirichlet(0.3 * np.ones(len(legal_probs)))
            noisy_probs = np.maximum(noisy_probs, 1e-12)
            noisy_probs = noisy_probs / np.sum(noisy_probs)
            move = int(self.rng.choice(acts, p=noisy_probs))
            self.update_with_move(move)
        else:
            legal_probs = np.maximum(legal_probs, 1e-12)
            legal_probs = legal_probs / np.sum(legal_probs)
            move = int(self.rng.choice(acts, p=legal_probs))

        if return_prob:
            return move, move_probs
        return move


class TrainPipeline:
    def __init__(self, board_size=11, n_in_row=5, fresh_start=False):
        # 基础参数
        self.board_width, self.board_height, self.n_in_row = int(board_size), int(board_size), int(n_in_row)
        if self.board_width < self.n_in_row:
            raise ValueError("board_size 不能小于 n_in_row")
        self.board_area = self.board_width * self.board_height
        self.fresh_start = bool(fresh_start)
        self.board = Board(self.board_width, self.board_height, self.n_in_row)
        self.game = Game(self.board)

        # 训练超参
        self.learn_rate = 2.0e-3
        self.lr_multiplier = 1.0
        self.temp = 1.0
        self.seed = 42
        self.cpu_count = max(1, multiprocessing.cpu_count())
        self.n_playout = 400
        self.c_puct = 5
        self.selfplay_backend = "cpp"
        self.eval_backend = "cpp"
        # 默认线程数偏稳健，避免高线程争用吞掉有效吞吐
        self.cpp_threads = max(4, min(16, self.cpu_count // 8))
        self.torch_cpu_threads = 1
        self.torch_interop_threads = 1
        self.batch_size = 512
        self.buffer_size = 100000
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.selfplay_async_enabled = True
        self.selfplay_worker_count = 2
        self.selfplay_prefetch_games = 10
        self.selfplay_queue_timeout_sec = 30.0
        self.epochs = 5
        self.kl_targ = 0.02
        self.check_freq = 50
        self.game_batch_num = 2000
        self.best_win_ratio = 0.0
        self.pure_mcts_playout_num = 1000
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.opening_temp_moves = 6
        self.opening_temp = 1.0
        self.midgame_temp = 0.35
        self.endgame_temp = 1e-3
        self.kl_explosion_threshold = self.kl_targ * 8
        self.high_kl_patience = 20
        self.eval_decline_patience = 30
        self.nan_retry_limit = 1
        self.model_dir = os.path.join(ROOT_DIR, "models")
        self.log_dir = os.path.join(ROOT_DIR, "runs")  # TensorBoard 根目录
        self.current_model_path = os.path.join(self.model_dir, "current_policy.pth")
        self.best_model_path = os.path.join(self.model_dir, "best_policy.pth")
        self.healthy_model_path = os.path.join(self.model_dir, "healthy_policy.pth")
        run_name = datetime.now().strftime(f"gomoku_{self.board_width}x{self.board_height}_%Y%m%d_%H%M%S")  # 每次训练单独子目录
        self.tb_run_dir = os.path.join(self.log_dir, run_name)
        self.train_log_path = os.path.join(self.tb_run_dir, "train.log")
        self.config_snapshot_path = os.path.join(self.tb_run_dir, "config_snapshot.json")
        self.eval_csv_path = os.path.join(self.tb_run_dir, "eval_metrics.csv")
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.tb_run_dir, exist_ok=True)
        self.logger = self._init_logger()
        self.writer = SummaryWriter(log_dir=self.tb_run_dir)  # 初始化 TensorBoard writer
        self._set_seed(self.seed)
        self._apply_torch_thread_settings()
        self._save_config_snapshot()
        self._init_eval_csv()

        # 网络与优化器
        self.policy_value_net = PolicyValueNet(self.board_width, num_channels=64, device=self.device)
        params, _ = self.policy_value_net.get_all_params()
        self.optimizer = Adam(params, lr=self.learn_rate)

        # MCTS 玩家
        self.use_cpp_selfplay = self.selfplay_backend == "cpp"
        self._net_lock = threading.Lock()  # protects NN forward/backward from concurrent access
        self.mcts_player = None
        self._build_selfplay_player(log_prefix="初始化")
        self.episode_len = 0
        self.train_step = 0      # 训练更新步（用于 train/eval 曲线横轴）
        self.selfplay_step = 0   # 自对弈局数（用于 selfplay 曲线横轴）
        self.last_kl = 0.0
        self.high_kl_streak = 0
        self.eval_win_history = deque(maxlen=self.eval_decline_patience)
        self.training_start_time = time.time()
        self.stop_requested = False
        self.stop_reason = ""
        self._eval_cpp_signature = None
        self.eval_cpp_board = None
        self.eval_cpp_current_player = None
        self.eval_cpp_pure_player = None
        self.selfplay_queue = None
        self.selfplay_stop_event = threading.Event()
        self.selfplay_workers = []

        if self.fresh_start:
            self._save_model(self.healthy_model_path, include_buffer=False)
            self._log("已启用 fresh-start，忽略历史断点并从零训练")
        else:
            resumed = self._load_model(self.current_model_path, restore_training_state=True)
            if resumed:
                self._log(f"已从断点恢复训练: {self.current_model_path}")
            else:
                self._save_model(self.healthy_model_path, include_buffer=False)
                self._log("未发现断点，开始全新训练")

    def _init_logger(self):
        logger = logging.getLogger(f"train_pipeline_{os.path.basename(self.tb_run_dir)}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

        file_handler = logging.FileHandler(self.train_log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)

        logger.handlers.clear()
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        return logger

    def _log(self, message):
        self.logger.info(message)

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _apply_torch_thread_settings(self):
        # GPU 训练时限制 torch CPU 线程，避免与 C++ 自对弈线程互抢
        torch.set_num_threads(max(1, int(self.torch_cpu_threads)))
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(max(1, int(self.torch_interop_threads)))
            except RuntimeError:
                # 该设置在某些运行时只允许初始化前调用，忽略即可
                pass

    def _make_cpp_eval_callback(self):
        """Create a batch evaluation callback for C++ MCTS neural network integration."""
        net = self.policy_value_net
        device = self.device
        board_area = self.board_area
        lock = self._net_lock

        def batch_eval(state_np, batch_size, width, height):
            # state_np: numpy array (batch_size, 4, width, height) from C++
            state_tensor = torch.from_numpy(
                np.ascontiguousarray(state_np)
            ).float().to(device)

            with lock:
                net.eval_mode()
                with torch.no_grad():
                    act_probs, values = net.forward(state_tensor)

            act_probs_np = act_probs.cpu().numpy()
            values_np = values.cpu().numpy()

            # Trim / pad policy dimension
            if act_probs_np.shape[1] > board_area:
                act_probs_np = act_probs_np[:, :board_area]
            elif act_probs_np.shape[1] < board_area:
                padded = np.zeros((batch_size, board_area), dtype=np.float32)
                padded[:, :act_probs_np.shape[1]] = act_probs_np
                act_probs_np = padded

            results = []
            for i in range(batch_size):
                results.append((act_probs_np[i].tolist(), float(values_np[i][0])))
            return results

        return batch_eval

    def _build_selfplay_player(self, log_prefix=""):
        self.use_cpp_selfplay = self.selfplay_backend == "cpp"
        if self.use_cpp_selfplay:
            eval_cb = self._make_cpp_eval_callback()
            self.mcts_player = CppSelfPlayPlayer(
                board_size=self.board_width,
                n_in_row=self.n_in_row,
                c_puct=self.c_puct,
                n_playout=self.n_playout,
                num_threads=self.cpp_threads,
                is_selfplay=True,
                seed=self.seed,
                eval_callback=eval_cb,
                eval_batch_size=getattr(self, 'eval_batch_size', 8),
            )
            self._log(
                f"{log_prefix}自对弈后端: C++ NN-MCTS "
                f"(threads={self.cpp_threads}, n_playout={self.n_playout}, "
                f"eval_batch_size={getattr(self, 'eval_batch_size', 8)}, cpu_count={self.cpu_count})"
            )
        else:
            self.mcts_player = MCTSPlayer(self.policy_value_fn, self.c_puct, self.n_playout, is_selfplay=1)
            self._log(f"{log_prefix}自对弈后端: Python MCTS (n_playout={self.n_playout})")

    def _start_selfplay_worker(self):
        if not self.selfplay_async_enabled:
            return
        if self.selfplay_workers:
            return

        if self.use_cpp_selfplay:
            worker_count = max(1, int(self.selfplay_worker_count))
        else:
            worker_count = 1

        maxsize = max(worker_count + 1, int(self.selfplay_prefetch_games))
        self.selfplay_queue = queue.Queue(maxsize=maxsize)
        self.selfplay_stop_event.clear()

        def _worker_loop(worker_id):
            worker_board, worker_player, use_cpp = self._make_selfplay_worker_components(worker_id)
            while not self.selfplay_stop_event.is_set():
                try:
                    item = self._collect_one_selfplay_game_payload_with_components(worker_board, worker_player, use_cpp)
                    self.selfplay_queue.put(item, timeout=0.5)
                except queue.Full:
                    continue
                except Exception as exc:
                    # 将异常透传给主线程处理
                    try:
                        self.selfplay_queue.put(("error", f"worker#{worker_id}: {exc}"), timeout=0.5)
                    except queue.Full:
                        pass
                    return

        self.selfplay_workers = []
        for wid in range(worker_count):
            t = threading.Thread(
                target=lambda i=wid: _worker_loop(i),
                name=f"selfplay-producer-{wid}",
                daemon=True,
            )
            t.start()
            self.selfplay_workers.append(t)
        self._log(f"已启动自对弈后台产数线程，workers={worker_count}, 队列容量={maxsize}")

    def _stop_selfplay_worker(self):
        self.selfplay_stop_event.set()
        for t in self.selfplay_workers:
            t.join(timeout=2.0)
        self.selfplay_workers = []

    def _make_selfplay_worker_components(self, worker_id):
        worker_board = Board(self.board_width, self.board_height, self.n_in_row)
        if self.use_cpp_selfplay:
            eval_cb = self._make_cpp_eval_callback()
            worker_player = CppSelfPlayPlayer(
                board_size=self.board_width,
                n_in_row=self.n_in_row,
                c_puct=self.c_puct,
                n_playout=self.n_playout,
                num_threads=self.cpp_threads,
                is_selfplay=True,
                seed=self.seed + 10007 * (worker_id + 1),
                eval_callback=eval_cb,
                eval_batch_size=getattr(self, 'eval_batch_size', 8),
            )
            return worker_board, worker_player, True

        worker_player = MCTSPlayer(self.policy_value_fn, self.c_puct, self.n_playout, is_selfplay=1)
        return worker_board, worker_player, False

    def _collect_one_selfplay_game_payload_with_components(self, board_obj, player_obj, use_cpp):
        board_obj.init_board()
        if use_cpp:
            player_obj.set_player_ind(1)
        player_obj.reset_player()

        states, mcts_probs, current_players = [], [], []
        temps = []
        step_times_ms = []

        while True:
            dynamic_temp = self._get_dynamic_temp(len(states))
            t0 = time.perf_counter()
            move, move_probs = player_obj.get_action(board_obj, temp=dynamic_temp, return_prob=1)
            step_times_ms.append((time.perf_counter() - t0) * 1000.0)

            states.append(board_obj.current_state())
            mcts_probs.append(move_probs)
            current_players.append(board_obj.current_player)
            temps.append(dynamic_temp)

            board_obj.do_move(move)
            is_end, winner = board_obj.game_end()
            if is_end:
                winners_z = np.zeros(len(current_players), dtype=np.float32)
                if winner != -1:
                    cp = np.array(current_players)
                    winners_z[cp == winner] = 1.0
                    winners_z[cp != winner] = -1.0

                play_data = list(zip(states, mcts_probs, winners_z))
                augmented = self.get_equi_data(play_data)
                avg_temp = float(np.mean(temps)) if temps else 0.0
                avg_ms = float(np.mean(step_times_ms)) if step_times_ms else 0.0
                return {
                    "episode_len": len(play_data),
                    "augmented_data": augmented,
                    "avg_temp": avg_temp,
                    "avg_step_ms": avg_ms,
                }

    def _collect_one_selfplay_game_payload(self):
        return self._collect_one_selfplay_game_payload_with_components(
            self.board,
            self.mcts_player,
            self.use_cpp_selfplay,
        )

    def _benchmark_cpp_threads(self, threads, tune_playout=128, tune_steps=12):
        bench_board = Board(self.board_width, self.board_height, self.n_in_row)
        bench_board.init_board()

        bench_player = CppSelfPlayPlayer(
            board_size=self.board_width,
            n_in_row=self.n_in_row,
            c_puct=self.c_puct,
            n_playout=tune_playout,
            num_threads=threads,
            is_selfplay=False,
        )
        bench_player.set_player_ind(1)
        bench_player.reset_player()

        elapsed_ms = []
        for _ in range(max(1, tune_steps)):
            if len(bench_board.availables) == 0:
                break
            t0 = time.perf_counter()
            move = bench_player.get_action(bench_board, temp=1e-3, return_prob=0)
            elapsed_ms.append((time.perf_counter() - t0) * 1000.0)
            bench_board.do_move(move)
            bench_player.update_with_move(move)

        if not elapsed_ms:
            return float("inf"), 0.0
        avg_ms = float(np.mean(elapsed_ms))
        pps = tune_playout / (avg_ms / 1000.0) if avg_ms > 0 else 0.0
        return avg_ms, pps

    def _auto_tune_cpp_threads(self, candidates=None, tune_playout=128, tune_steps=12):
        if candidates is None:
            default_candidates = [8, 12, 16, 24, 32, 48, 64, 96, 128]
            candidates = [x for x in default_candidates if x <= self.cpu_count]
            if not candidates:
                candidates = [max(1, self.cpu_count)]

        # 去重并过滤非法值
        filtered = []
        seen = set()
        for x in candidates:
            xv = int(x)
            if xv <= 0:
                continue
            if xv > self.cpu_count:
                xv = self.cpu_count
            if xv not in seen:
                seen.add(xv)
                filtered.append(xv)

        if not filtered:
            filtered = [max(1, self.cpu_count)]

        self._log(
            f"开始自动寻优 cpp_threads: candidates={filtered}, "
            f"tune_playout={tune_playout}, tune_steps={tune_steps}"
        )

        best_threads = filtered[0]
        best_pps = -1.0
        best_ms = float("inf")
        for t in filtered:
            avg_ms, pps = self._benchmark_cpp_threads(t, tune_playout=tune_playout, tune_steps=tune_steps)
            self._log(f"[THREAD TUNE] threads={t}, avg_ms={avg_ms:.3f}, pps={pps:.1f}")
            if pps > best_pps:
                best_pps = pps
                best_ms = avg_ms
                best_threads = t

        self.cpp_threads = best_threads
        self._eval_cpp_signature = None
        self._log(
            f"自动寻优完成: best_threads={best_threads}, "
            f"avg_ms={best_ms:.3f}, pps={best_pps:.1f}"
        )

    def _build_eval_cpp_players(self):
        signature = (
            self.board_width,
            self.board_height,
            self.n_in_row,
            int(self.c_puct),
            int(self.n_playout),
            int(self.pure_mcts_playout_num),
            int(self.cpp_threads),
        )
        if self._eval_cpp_signature == signature and self.eval_cpp_board is not None:
            return

        mcts_cpp = _import_mcts_cpp_module()
        self.eval_cpp_board = mcts_cpp.Board(self.board_width, self.board_height, self.n_in_row)

        # Current player: uses neural network for evaluation
        self.eval_cpp_current_player = mcts_cpp.AlphaZeroPlayer(
            c_puct=int(self.c_puct),
            n_playout=int(self.n_playout),
            num_threads=int(self.cpp_threads),
        )
        eval_cb = self._make_cpp_eval_callback()
        self.eval_cpp_current_player.set_eval_callback(
            eval_cb, getattr(self, 'eval_batch_size', 8)
        )

        # Pure MCTS opponent: NO neural network (pure rollout baseline)
        self.eval_cpp_pure_player = mcts_cpp.AlphaZeroPlayer(
            c_puct=5,
            n_playout=int(self.pure_mcts_playout_num),
            num_threads=int(self.cpp_threads),
        )
        self.eval_cpp_current_player.set_player_ind(1)
        self.eval_cpp_pure_player.set_player_ind(2)
        self._eval_cpp_signature = signature

    def _save_config_snapshot(self):
        config = {
            "board_width": self.board_width,
            "board_height": self.board_height,
            "n_in_row": self.n_in_row,
            "learn_rate": self.learn_rate,
            "lr_multiplier": self.lr_multiplier,
            "batch_size": self.batch_size,
            "buffer_size": self.buffer_size,
            "epochs": self.epochs,
            "kl_targ": self.kl_targ,
            "n_playout": self.n_playout,
            "c_puct": self.c_puct,
            "selfplay_backend": self.selfplay_backend,
            "eval_backend": self.eval_backend,
            "cpp_threads": self.cpp_threads,
            "selfplay_async_enabled": self.selfplay_async_enabled,
            "selfplay_worker_count": self.selfplay_worker_count,
            "selfplay_prefetch_games": self.selfplay_prefetch_games,
            "torch_cpu_threads": self.torch_cpu_threads,
            "torch_interop_threads": self.torch_interop_threads,
            "check_freq": self.check_freq,
            "game_batch_num": self.game_batch_num,
            "seed": self.seed,
            "opening_temp_moves": self.opening_temp_moves,
            "opening_temp": self.opening_temp,
            "endgame_temp": self.endgame_temp,
            "device": self.device,
            "created_at": datetime.now().isoformat(),
        }
        with open(self.config_snapshot_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    def _init_eval_csv(self):
        if os.path.exists(self.eval_csv_path):
            return
        with open(self.eval_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["train_step", "pure_mcts_playout", "win_ratio", "elapsed_sec", "kl"])

    def _append_eval_csv(self, win_ratio):
        elapsed = time.time() - self.training_start_time
        with open(self.eval_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.train_step,
                self.pure_mcts_playout_num,
                f"{win_ratio:.6f}",
                f"{elapsed:.3f}",
                f"{self.last_kl:.6f}",
            ])

    def policy_value_fn(self, board):
        # 桥接层：环境态 -> MCTS 概率
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(-1, 4, self.board_width, self.board_height))
        state_tensor = torch.tensor(current_state, dtype=torch.float32, device=self.device)

        self.policy_value_net.eval_mode()
        with torch.no_grad():
            act_probs_tensor, value_tensor = self.policy_value_net.forward(state_tensor)

        # 动态适配策略维度（兼容带/不带 pass 动作）
        act_probs_full = act_probs_tensor.cpu().numpy()[0]
        if act_probs_full.shape[0] == self.board_area + 1:
            act_probs = act_probs_full[:-1]
        elif act_probs_full.shape[0] == self.board_area:
            act_probs = act_probs_full
        elif act_probs_full.shape[0] > self.board_area:
            act_probs = act_probs_full[:self.board_area]
        else:
            raise ValueError(
                f"策略输出维度不足: got={act_probs_full.shape[0]}, expected>={self.board_area}"
            )
        value = value_tensor.cpu().numpy()[0][0]

        if len(legal_positions) == 0:
            return [], value

        legal_probs = act_probs[legal_positions]
        prob_sum = np.sum(legal_probs)
        if prob_sum > 0:
            legal_probs = legal_probs / prob_sum
        else:
            legal_probs = np.full(len(legal_positions), 1.0 / len(legal_positions), dtype=np.float32)

        return zip(legal_positions, legal_probs), value

    def get_equi_data(self, play_data):
        # 数据增强：旋转翻转 (1局变8局)
        extend_data = []
        for state, mcts_prob, winner in play_data:
            for i in [1, 2, 3, 4]:
                s_rot = np.array([np.rot90(s, i) for s in state])
                p_rot = np.rot90(np.flipud(mcts_prob.reshape(self.board_height, self.board_width)), i)
                extend_data.append((s_rot, np.flipud(p_rot).flatten(), winner))

                s_flip = np.array([np.fliplr(s) for s in s_rot])
                p_flip = np.fliplr(p_rot)
                extend_data.append((s_flip, np.flipud(p_flip).flatten(), winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        # 自对弈产数（支持后台并行流水）
        if self.selfplay_async_enabled:
            self._start_selfplay_worker()

        for _ in range(n_games):
            if self.selfplay_async_enabled:
                item = self.selfplay_queue.get(timeout=self.selfplay_queue_timeout_sec)
                if isinstance(item, tuple) and item and item[0] == "error":
                    raise RuntimeError(f"后台自对弈线程异常: {item[1]}")
                payload = item
            else:
                payload = self._collect_one_selfplay_game_payload()

            self.episode_len = int(payload["episode_len"])
            self.data_buffer.extend(payload["augmented_data"])
            self.selfplay_step += 1

            self.writer.add_scalar("selfplay/episode_len", self.episode_len, self.selfplay_step)
            self.writer.add_scalar("selfplay/buffer_size", len(self.data_buffer), self.selfplay_step)
            self.writer.add_scalar("selfplay/avg_temp", float(payload["avg_temp"]), self.selfplay_step)
            avg_ms = float(payload["avg_step_ms"])
            self.writer.add_scalar("selfplay/avg_step_ms", avg_ms, self.selfplay_step)
            if avg_ms > 0:
                self.writer.add_scalar("selfplay/playout_per_sec", self.n_playout / (avg_ms / 1000.0), self.selfplay_step)

    def _get_dynamic_temp(self, move_idx):
        if self.board_width == 8 and self.board_height == 8:
            return self.opening_temp if move_idx < self.opening_temp_moves else self.endgame_temp
        return self.temp

    def _compute_grad_norm(self):
        params, _ = self.policy_value_net.get_all_params()
        norm_sq = 0.0
        for p in params.values():
            if p is not None and p.grad is not None:
                norm_sq += torch.sum(p.grad.detach() ** 2).item()
        return norm_sq ** 0.5

    def _reduce_lr_on_failure(self):
        self.lr_multiplier = max(self.lr_multiplier / 2.0, 0.05)
        self.optimizer.lr = self.learn_rate * self.lr_multiplier

    def _rollback_to_healthy_checkpoint(self):
        for path in [self.healthy_model_path, self.best_model_path, self.current_model_path]:
            if self._load_model(path, restore_training_state=False):
                self._log(f"已回滚到健康模型: {path}")
                return True
        self._log("未找到可回滚模型，继续当前参数")
        return False

    def policy_update(self):
        # 采样与网络权重更新（带 NaN/爆炸熔断 + 回滚）
        for attempt in range(self.nan_retry_limit + 1):
            try:
                mini_batch = random.sample(self.data_buffer, self.batch_size)
                state_batch = torch.tensor(np.array([d[0] for d in mini_batch]), dtype=torch.float32, device=self.device)
                mcts_probs_batch = torch.tensor(np.array([d[1] for d in mini_batch]), dtype=torch.float32, device=self.device)
                winner_batch = torch.tensor(np.array([d[2] for d in mini_batch]), dtype=torch.float32, device=self.device).unsqueeze(1)

                # ── 大锁：整个 forward/backward/KL 过程独占网络 ──
                with self._net_lock:
                    self.policy_value_net.train_mode()
                    with torch.no_grad():
                        old_probs, old_v = self.policy_value_net.forward(state_batch)
                    old_probs = old_probs.detach()
                    old_v = old_v.detach()
                    kl = 0.0

                    for _ in range(self.epochs):
                        self.policy_value_net.zero_grad()
                        act_probs, value = self.policy_value_net.forward(state_batch)

                        # 动态对齐 MCTS 目标与网络输出维度
                        if act_probs.shape[1] == mcts_probs_batch.shape[1] + 1:
                            target_policy_full = torch.zeros_like(act_probs)
                            target_policy_full[:, :-1] = mcts_probs_batch
                            entropy_probs = act_probs[:, :-1]
                        elif act_probs.shape[1] == mcts_probs_batch.shape[1]:
                            target_policy_full = mcts_probs_batch
                            entropy_probs = act_probs
                        else:
                            raise FloatingPointError(
                                f"策略维度不匹配: net={act_probs.shape[1]}, target={mcts_probs_batch.shape[1]}"
                            )

                        # 手算 Loss 和 梯度（与网络输出维度对齐）
                        total_loss, loss_v, loss_p, grad_v, grad_p = combined_loss(
                            act_probs, value, target_policy_full, winner_batch
                        )

                        if not torch.isfinite(total_loss):
                            raise FloatingPointError("Loss 出现 NaN/Inf")

                        policy_entropy = torch.mean(
                            -torch.sum(entropy_probs * torch.log(entropy_probs + 1e-10), dim=1)
                        )

                        # 手动反向传播与优化
                        self.policy_value_net.backward(grad_p, grad_v)
                        grad_norm = self._compute_grad_norm()
                        if not np.isfinite(grad_norm):
                            raise FloatingPointError("grad_norm 出现 NaN/Inf")
                        self.optimizer.step()

                        with torch.no_grad():
                            new_probs, _ = self.policy_value_net.forward(state_batch)
                            kl = torch.mean(
                                torch.sum(
                                    old_probs * (torch.log(old_probs + 1e-10) - torch.log(new_probs + 1e-10)),
                                    dim=1
                                )
                            ).item()

                        if not np.isfinite(kl) or kl > self.kl_explosion_threshold:
                            raise FloatingPointError(f"KL 爆炸: {kl:.5f}")

                        if kl > self.kl_targ * 4:
                            break
                # ── 大锁结束 ──

                if kl > self.kl_targ * 1.8 and self.lr_multiplier > 0.2:
                    self.lr_multiplier /= 1.25
                elif kl < self.kl_targ * 0.7 and self.lr_multiplier < 1.5:
                    self.lr_multiplier *= 1.1
                self.lr_multiplier = float(np.clip(self.lr_multiplier, 0.2, 1.5))
                self.optimizer.lr = self.learn_rate * self.lr_multiplier

                explained_var_old = 1 - torch.var(winner_batch - old_v) / (torch.var(winner_batch) + 1e-10)
                explained_var_new = 1 - torch.var(winner_batch - value.detach()) / (torch.var(winner_batch) + 1e-10)
                self._log(
                    f"kl:{kl:.5f}, lr_mult:{self.lr_multiplier:.3f}, "
                    f"loss:{total_loss.item():.4f}, v_loss:{loss_v.item():.4f}, p_loss:{loss_p.item():.4f}, "
                    f"ev_old:{explained_var_old.item():.3f}, ev_new:{explained_var_new.item():.3f}, "
                    f"grad_norm:{grad_norm:.4f}, entropy:{policy_entropy.item():.4f}"
                )

                self.train_step += 1
                self.last_kl = kl
                if kl > self.kl_targ * 3:
                    self.high_kl_streak += 1
                else:
                    self.high_kl_streak = 0

                # 记录训练主指标
                self.writer.add_scalar("train/loss_total", total_loss.item(), self.train_step)
                self.writer.add_scalar("train/loss_value", loss_v.item(), self.train_step)
                self.writer.add_scalar("train/loss_policy", loss_p.item(), self.train_step)
                self.writer.add_scalar("train/kl", kl, self.train_step)
                self.writer.add_scalar("train/lr", self.optimizer.lr, self.train_step)
                self.writer.add_scalar("train/lr_multiplier", self.lr_multiplier, self.train_step)
                self.writer.add_scalar("train/explained_var_old", explained_var_old.item(), self.train_step)
                self.writer.add_scalar("train/explained_var_new", explained_var_new.item(), self.train_step)
                self.writer.add_scalar("train/grad_norm", grad_norm, self.train_step)
                self.writer.add_scalar("train/policy_entropy", policy_entropy.item(), self.train_step)

                self._save_model(self.healthy_model_path, include_buffer=False)
                return total_loss.item(), loss_v.item(), loss_p.item()

            except FloatingPointError as err:
                self._log(f"数值熔断触发: {err}. 丢弃当前 batch 并回滚重试。")
                self._rollback_to_healthy_checkpoint()
                self._reduce_lr_on_failure()
                if attempt >= self.nan_retry_limit:
                    self._log("重试次数耗尽，跳过本次更新")
                    self.high_kl_streak += 1
                    return None, None, None

        return None, None, None

    def _play_cpp_eval_game(self, start_player=0):
        """C++ 后端评估：当前引擎(较高 playout) vs 基线引擎(较低 playout)。"""
        self._build_eval_cpp_players()
        c_board = self.eval_cpp_board
        current_player = self.eval_cpp_current_player
        pure_player = self.eval_cpp_pure_player

        c_board.init_board(start_player=start_player)
        current_player.reset_player()
        pure_player.reset_player()
        players = {1: current_player, 2: pure_player}

        while True:
            current_turn = c_board.current_player
            player_in_turn = players[current_turn]
            if c_board.last_move != -1:
                player_in_turn.update_with_move(c_board.last_move)

            move = player_in_turn.get_action(c_board)
            c_board.do_move(move)
            # 双方同步最新一步，确保树状态一致
            current_player.update_with_move(move)
            pure_player.update_with_move(move)

            is_end, winner = c_board.game_end()
            if is_end:
                return winner

    def _policy_evaluate_python(self, n_games=10):
        # 与纯 MCTS 对战评估（Python 后端）
        t0 = time.perf_counter()
        current_mcts_player = MCTSPlayer(
            self.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout, is_selfplay=0
        )
        pure_mcts_player = MCTS_Pure(c_puct=5, n_playout=self.pure_mcts_playout_num)

        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(
                current_mcts_player, pure_mcts_player, start_player=i % 2, is_shown=0
            )
            win_cnt[winner] += 1

        win_ratio = (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
        self._log(
            f"num_playouts:{self.pure_mcts_playout_num}, "
            f"win:{win_cnt[1]}, lose:{win_cnt[2]}, tie:{win_cnt[-1]}"
        )
        eval_elapsed = time.perf_counter() - t0
        self._log(f"[PY EVAL] elapsed_sec:{eval_elapsed:.3f}")
        # 记录评估胜率与当前评估强度
        self.writer.add_scalar("eval/win_ratio", win_ratio, self.train_step)
        self.writer.add_scalar("eval/pure_mcts_playout_num", self.pure_mcts_playout_num, self.train_step)
        self.writer.add_scalar("eval/elapsed_sec", eval_elapsed, self.train_step)
        self._append_eval_csv(win_ratio)
        return win_ratio

    def _policy_evaluate_cpp(self, n_games=10):
        # C++ 后端评估：当前引擎 vs 纯基线引擎
        t0 = time.perf_counter()
        self._build_eval_cpp_players()
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self._play_cpp_eval_game(start_player=i % 2)
            win_cnt[winner] += 1

        # 约定 player1 为 current 引擎，player2 为 pure 引擎
        win_ratio = (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
        self._log(
            f"[CPP EVAL] num_playouts:{self.pure_mcts_playout_num}, "
            f"win:{win_cnt[1]}, lose:{win_cnt[2]}, tie:{win_cnt[-1]}"
        )
        eval_elapsed = time.perf_counter() - t0
        self._log(f"[CPP EVAL] elapsed_sec:{eval_elapsed:.3f}")
        self.writer.add_scalar("eval/win_ratio", win_ratio, self.train_step)
        self.writer.add_scalar("eval/pure_mcts_playout_num", self.pure_mcts_playout_num, self.train_step)
        self.writer.add_scalar("eval/elapsed_sec", eval_elapsed, self.train_step)
        self._append_eval_csv(win_ratio)
        return win_ratio

    def policy_evaluate(self, n_games=10):
        if self.eval_backend == "cpp":
            return self._policy_evaluate_cpp(n_games=n_games)
        return self._policy_evaluate_python(n_games=n_games)

    def _capture_model_state(self):
        params, _ = self.policy_value_net.get_all_params()
        return {k: v.detach().cpu().clone() for k, v in params.items() if v is not None}

    def _restore_model_state(self, model_state):
        params, _ = self.policy_value_net.get_all_params()
        loaded_keys = 0
        for name, tensor in model_state.items():
            if name in params and params[name] is not None:
                if tuple(params[name].shape) == tuple(tensor.shape):
                    params[name].data.copy_(tensor.to(params[name].device))
                    loaded_keys += 1
        return loaded_keys

    def _capture_optimizer_state(self):
        state = {}
        for name, value in self.optimizer.state.items():
            if isinstance(value, dict):
                state[name] = {
                    k: (v.detach().cpu().clone() if torch.is_tensor(v) else v)
                    for k, v in value.items()
                }
            else:
                state[name] = value.detach().cpu().clone() if torch.is_tensor(value) else value
        return {
            "state": state,
            "lr": self.optimizer.lr,
            "t": getattr(self.optimizer, "t", None),
        }

    def _restore_optimizer_state(self, optimizer_state):
        if not optimizer_state:
            return
        self.optimizer.lr = optimizer_state.get("lr", self.optimizer.lr)
        if hasattr(self.optimizer, "t") and optimizer_state.get("t") is not None:
            self.optimizer.t = optimizer_state["t"]
        state = optimizer_state.get("state", {})
        for name, value in state.items():
            if isinstance(value, dict):
                if name not in self.optimizer.state or not isinstance(self.optimizer.state[name], dict):
                    self.optimizer.state[name] = {}
                for k, v in value.items():
                    self.optimizer.state[name][k] = v.to(self.device) if torch.is_tensor(v) else v
            else:
                self.optimizer.state[name] = value.to(self.device) if torch.is_tensor(value) else value

    def _save_model(self, path, include_buffer=True):
        checkpoint = {
            "model_state": self._capture_model_state(),
            "optimizer_state": self._capture_optimizer_state(),
            "best_win_ratio": self.best_win_ratio,
            "pure_mcts_playout_num": self.pure_mcts_playout_num,
            "lr_multiplier": self.lr_multiplier,
            "train_step": self.train_step,
            "selfplay_step": self.selfplay_step,
            "last_kl": self.last_kl,
            "high_kl_streak": self.high_kl_streak,
            "eval_win_history": list(self.eval_win_history),
            "data_buffer": list(self.data_buffer) if include_buffer else None,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        tmp_path = f"{path}.tmp"
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path)

    def _load_model(self, path, restore_training_state=True):
        if not os.path.exists(path):
            return False
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if "model_state" not in checkpoint:
            return False
        loaded_keys = self._restore_model_state(checkpoint["model_state"])
        if loaded_keys == 0:
            # 常见于 8x8 checkpoint 误加载到 11x11 模型，直接按未命中处理
            return False
        self._restore_optimizer_state(checkpoint.get("optimizer_state"))
        if not restore_training_state:
            return True

        self.best_win_ratio = checkpoint.get("best_win_ratio", self.best_win_ratio)
        self.pure_mcts_playout_num = checkpoint.get("pure_mcts_playout_num", self.pure_mcts_playout_num)
        self.lr_multiplier = float(np.clip(checkpoint.get("lr_multiplier", self.lr_multiplier), 0.2, 1.5))
        self.optimizer.lr = self.learn_rate * self.lr_multiplier
        self.train_step = checkpoint.get("train_step", self.train_step)
        self.selfplay_step = checkpoint.get("selfplay_step", self.selfplay_step)
        self.last_kl = checkpoint.get("last_kl", self.last_kl)
        self.high_kl_streak = checkpoint.get("high_kl_streak", self.high_kl_streak)
        self.eval_win_history = deque(
            checkpoint.get("eval_win_history", []),
            maxlen=self.eval_decline_patience
        )

        buffer_data = checkpoint.get("data_buffer")
        if buffer_data is not None:
            self.data_buffer = deque(buffer_data, maxlen=self.buffer_size)

        if checkpoint.get("python_rng_state") is not None:
            random.setstate(checkpoint["python_rng_state"])
        if checkpoint.get("numpy_rng_state") is not None:
            np.random.set_state(checkpoint["numpy_rng_state"])
        if checkpoint.get("torch_rng_state") is not None:
            torch.set_rng_state(checkpoint["torch_rng_state"])
        if torch.cuda.is_available() and checkpoint.get("torch_cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(checkpoint["torch_cuda_rng_state"])
        return True

    def _should_soft_stop(self):
        if self.high_kl_streak >= self.high_kl_patience:
            self.stop_reason = f"KL 长期过高（连续 {self.high_kl_streak} 次）"
            return True
        if len(self.eval_win_history) >= self.eval_decline_patience:
            history = list(self.eval_win_history)
            if all(history[i] > history[i + 1] for i in range(len(history) - 1)):
                self.stop_reason = f"评估胜率连续 {self.eval_decline_patience} 次下降"
                return True
        return False

    def _handle_soft_stop(self):
        self._log(f"触发软早停: {self.stop_reason}")
        if os.path.exists(self.best_model_path):
            self._load_model(self.best_model_path, restore_training_state=False)
            self._log("已回滚到 best_policy.pth")
        self._reduce_lr_on_failure()
        self._save_model(self.current_model_path)
        self.stop_requested = True

    def run(self):
        # 主循环
        import itertools
        self._log("训练开始")
        if self.selfplay_async_enabled:
            self._start_selfplay_worker()
        try:
            iterator = range(self.game_batch_num) if self.game_batch_num > 0 else itertools.count()
            for i in iterator:
                self.collect_selfplay_data(self.play_batch_size)
                self._log(f"Batch {i+1}: 游戏步数 {self.episode_len}, Buffer容量 {len(self.data_buffer)}")

                if len(self.data_buffer) >= self.batch_size:
                    loss, loss_v, loss_p = self.policy_update()
                    if loss is not None:
                        self._log(f"更新完毕 | Loss: {loss:.4f} (V: {loss_v:.4f}, P: {loss_p:.4f})")

                if (i + 1) % self.check_freq == 0:
                    self._log(f"current self-play batch: {i+1}")
                    win_ratio = self.policy_evaluate()
                    self.eval_win_history.append(win_ratio)
                    self._save_model(self.current_model_path)
                    if win_ratio > self.best_win_ratio:
                        self._log("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        self._save_model(self.best_model_path)
                        if self.best_win_ratio == 1.0 and self.pure_mcts_playout_num < 5000:
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0

                if self._should_soft_stop():
                    self._handle_soft_stop()
                    break
                self.writer.add_scalar("train/batch_index", i + 1, i + 1)
                self.writer.add_scalar("train/high_kl_streak", self.high_kl_streak, i + 1)

        except KeyboardInterrupt:
            self._log("训练终止（KeyboardInterrupt）")
        finally:
            self._stop_selfplay_worker()
            if self.stop_requested:
                self._log(f"训练已软停止: {self.stop_reason}")
            # 确保日志落盘，避免中断导致 TensorBoard 数据丢失
            self.writer.flush()
            self.writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train pipeline with optional C++ selfplay backend")
    parser.add_argument("--board-size", type=int, default=11, help="棋盘边长 N（n*n）")
    parser.add_argument("--n-in-row", type=int, default=5, help="连珠数")
    parser.add_argument("--fresh-start", action="store_true", help="忽略已有断点，从零开始训练")
    parser.add_argument("--selfplay-backend", choices=["cpp", "python"], default="cpp", help="自对弈搜索后端")
    parser.add_argument("--eval-backend", choices=["cpp", "python"], default="cpp", help="评估搜索后端")
    parser.add_argument("--cpp-threads", type=int, default=None, help="C++ 自对弈线程数，默认自动按 CPU 核心估算")
    parser.add_argument("--n-playout", type=int, default=None, help="覆盖每步模拟次数")
    parser.add_argument("--torch-cpu-threads", type=int, default=None, help="训练进程 PyTorch CPU 线程数")
    parser.add_argument("--torch-interop-threads", type=int, default=None, help="训练进程 PyTorch 互操作线程数")
    parser.add_argument("--auto-tune-threads", action="store_true", help="启动前自动寻优 cpp_threads")
    parser.add_argument("--cpp-thread-candidates", type=str, default=None, help="线程候选列表，如 16,24,32,48,64")
    parser.add_argument("--tune-playout", type=int, default=128, help="线程寻优时每步 playout")
    parser.add_argument("--tune-steps", type=int, default=12, help="线程寻优时测试步数")
    parser.add_argument("--disable-selfplay-async", action="store_true", help="关闭后台并行产数流水")
    parser.add_argument("--selfplay-workers", type=int, default=2, help="后台产数 worker 数量")
    parser.add_argument("--selfplay-prefetch-games", type=int, default=6, help="后台产数队列容量（局数）")
    parser.add_argument("--selfplay-queue-timeout", type=float, default=30.0, help="主线程取产数超时秒数")
    parser.add_argument("--game-batch-num", type=int, default=None, help="训练总轮数，0 表示无限循环")
    parser.add_argument("--eval-batch-size", type=int, default=8, help="C++ MCTS 神经网络批量推理大小")
    args = parser.parse_args()

    pipeline = TrainPipeline(
        board_size=args.board_size,
        n_in_row=args.n_in_row,
        fresh_start=args.fresh_start,
    )
    if args.n_playout is not None:
        pipeline.n_playout = int(args.n_playout)
    pipeline.selfplay_backend = args.selfplay_backend
    pipeline.eval_backend = args.eval_backend
    if args.cpp_threads is not None:
        pipeline.cpp_threads = max(1, int(args.cpp_threads))
    if args.torch_cpu_threads is not None:
        pipeline.torch_cpu_threads = max(1, int(args.torch_cpu_threads))
    if args.torch_interop_threads is not None:
        pipeline.torch_interop_threads = max(1, int(args.torch_interop_threads))
    if args.disable_selfplay_async:
        pipeline.selfplay_async_enabled = False
    pipeline.selfplay_worker_count = max(1, int(args.selfplay_workers))
    pipeline.selfplay_prefetch_games = max(2, int(args.selfplay_prefetch_games))
    pipeline.selfplay_queue_timeout_sec = max(1.0, float(args.selfplay_queue_timeout))
    if args.game_batch_num is not None:
        pipeline.game_batch_num = int(args.game_batch_num)
    pipeline.eval_batch_size = max(1, int(args.eval_batch_size))

    pipeline._apply_torch_thread_settings()

    if args.auto_tune_threads and (pipeline.selfplay_backend == "cpp" or pipeline.eval_backend == "cpp"):
        candidates = None
        if args.cpp_thread_candidates:
            candidates = [int(x.strip()) for x in args.cpp_thread_candidates.split(",") if x.strip()]
        pipeline._auto_tune_cpp_threads(
            candidates=candidates,
            tune_playout=max(8, int(args.tune_playout)),
            tune_steps=max(2, int(args.tune_steps)),
        )

    pipeline._build_selfplay_player(log_prefix="CLI覆盖|")
    pipeline._log(f"CLI覆盖|评估后端: {pipeline.eval_backend}")

    pipeline.run()
