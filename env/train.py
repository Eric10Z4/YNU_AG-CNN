import csv
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict, deque
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from game import Board, Game
from mcts_alphaZero import MCTSPlayer
from mcts_pure import MCTSPlayer as MCTS_Pure

# 导入手搓组件
from pipeline.policy_value_net import PolicyValueNet
from pipeline.optimizer import Adam
from pipeline.losses import combined_loss


class TrainPipeline:
    def __init__(self):
        # 基础参数
        self.board_width, self.board_height, self.n_in_row = 15, 15, 5
        self.board = Board(self.board_width, self.board_height, self.n_in_row)
        self.game = Game(self.board)

        # 训练超参
        self.learn_rate = 1.5e-3
        self.lr_multiplier = 1.0
        self.temp = 1.0
        self.seed = 42
        self.n_playout = 200
        self.c_puct = 5
        self.batch_size = 256
        self.buffer_size = 10000
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 3
        self.kl_targ = 0.02
        self.check_freq = 50  # 每 50 次迭代评估一次模型
        self.game_batch_num = 1500
        self.best_win_ratio = 0.0
        self.pure_mcts_playout_num = 600
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.opening_temp_moves = 40
        self.opening_temp = 1.0
        self.endgame_temp = 1e-3
        self.kl_explosion_threshold = self.kl_targ * 8
        self.high_kl_patience = 10
        self.eval_decline_patience = 10
        self.nan_retry_limit = 1
        self.model_dir = os.path.join(ROOT_DIR, "models")
        self.log_dir = os.path.join(ROOT_DIR, "runs")  # TensorBoard 根目录
        self.current_model_path = os.path.join(self.model_dir, "current_policy.pth")
        self.best_model_path = os.path.join(self.model_dir, "best_policy.pth")
        self.healthy_model_path = os.path.join(self.model_dir, "healthy_policy.pth")
        run_name = datetime.now().strftime("gomoku_15x15_%Y%m%d_%H%M%S")  # 每次训练单独子目录
        self.tb_run_dir = os.path.join(self.log_dir, run_name)
        self.train_log_path = os.path.join(self.tb_run_dir, "train.log")
        self.config_snapshot_path = os.path.join(self.tb_run_dir, "config_snapshot.json")
        self.eval_csv_path = os.path.join(self.tb_run_dir, "eval_metrics.csv")
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.tb_run_dir, exist_ok=True)
        self.logger = self._init_logger()
        self.writer = SummaryWriter(log_dir=self.tb_run_dir)  # 初始化 TensorBoard writer
        self._set_seed(self.seed)
        self._save_config_snapshot()
        self._init_eval_csv()

        # 网络与优化器
        self.policy_value_net = PolicyValueNet(self.board_width, num_channels=64, device=self.device)
        params, _ = self.policy_value_net.get_all_params()
        self.optimizer = Adam(params, lr=self.learn_rate)

        # MCTS 玩家
        self.mcts_player = MCTSPlayer(self.policy_value_fn, self.c_puct, self.n_playout, is_selfplay=1)
        self.episode_len = 0
        self.train_step = 0      # 训练更新步（用于 train/eval 曲线横轴）
        self.selfplay_step = 0   # 自对弈局数（用于 selfplay 曲线横轴）
        self.last_kl = 0.0
        self.high_kl_streak = 0
        self.eval_win_history = deque(maxlen=self.eval_decline_patience)
        self.training_start_time = time.time()
        self.stop_requested = False
        self.stop_reason = ""

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

        # 切除第 226 维 (无用的 Pass 动作)
        act_probs = act_probs_tensor.cpu().numpy()[0][:-1]
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
        # 自对弈产数
        for _ in range(n_games):
            self.board.init_board()
            self.mcts_player.reset_player()
            states, mcts_probs, current_players = [], [], []
            temps = []

            while True:
                dynamic_temp = self._get_dynamic_temp(len(states))
                move, move_probs = self.mcts_player.get_action(self.board, temp=dynamic_temp, return_prob=1)
                states.append(self.board.current_state())
                mcts_probs.append(move_probs)
                current_players.append(self.board.current_player)
                temps.append(dynamic_temp)

                self.board.do_move(move)
                is_end, winner = self.board.game_end()

                if is_end:
                    winners_z = np.zeros(len(current_players))
                    if winner != -1:
                        winners_z[np.array(current_players) == winner] = 1.0
                        winners_z[np.array(current_players) != winner] = -1.0

                    play_data = list(zip(states, mcts_probs, winners_z))
                    self.episode_len = len(play_data)
                    self.data_buffer.extend(self.get_equi_data(play_data))
                    self.selfplay_step += 1
                    # 记录自对弈产数质量与数据池规模
                    self.writer.add_scalar("selfplay/episode_len", self.episode_len, self.selfplay_step)
                    self.writer.add_scalar("selfplay/buffer_size", len(self.data_buffer), self.selfplay_step)
                    self.writer.add_scalar("selfplay/avg_temp", float(np.mean(temps)), self.selfplay_step)
                    break

    def _get_dynamic_temp(self, move_idx):
        if self.board_width == 15 and self.board_height == 15:
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

                self.policy_value_net.train_mode()
                with torch.no_grad():
                    old_probs, old_v = self.policy_value_net.forward(state_batch)
                old_probs = old_probs.detach()
                old_v = old_v.detach()
                kl = 0.0

                for _ in range(self.epochs):
                    self.policy_value_net.zero_grad()
                    act_probs, value = self.policy_value_net.forward(state_batch)

                    # 将 225 维 MCTS 目标补齐到 226 维（pass 动作监督为 0）
                    target_policy_full = torch.zeros_like(act_probs)
                    target_policy_full[:, :-1] = mcts_probs_batch

                    # 手算 Loss 和 梯度（与网络 226 维输出对齐）
                    total_loss, loss_v, loss_p, grad_v, grad_p = combined_loss(
                        act_probs, value, target_policy_full, winner_batch
                    )

                    if not torch.isfinite(total_loss):
                        raise FloatingPointError("Loss 出现 NaN/Inf")

                    policy_entropy = torch.mean(
                        -torch.sum(act_probs[:, :-1] * torch.log(act_probs[:, :-1] + 1e-10), dim=1)
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

                if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
                    self.lr_multiplier /= 1.5
                elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
                    self.lr_multiplier *= 1.5
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

    def policy_evaluate(self, n_games=10):
        # 与纯 MCTS 对战评估
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
        # 记录评估胜率与当前评估强度
        self.writer.add_scalar("eval/win_ratio", win_ratio, self.train_step)
        self.writer.add_scalar("eval/pure_mcts_playout_num", self.pure_mcts_playout_num, self.train_step)
        self._append_eval_csv(win_ratio)
        return win_ratio

    def _capture_model_state(self):
        params, _ = self.policy_value_net.get_all_params()
        return {k: v.detach().cpu().clone() for k, v in params.items() if v is not None}

    def _restore_model_state(self, model_state):
        params, _ = self.policy_value_net.get_all_params()
        for name, tensor in model_state.items():
            if name in params and params[name] is not None:
                params[name].data.copy_(tensor.to(params[name].device))

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
        checkpoint = torch.load(path, map_location="cpu")
        if "model_state" not in checkpoint:
            return False
        self._restore_model_state(checkpoint["model_state"])
        self._restore_optimizer_state(checkpoint.get("optimizer_state"))
        if not restore_training_state:
            return True

        self.best_win_ratio = checkpoint.get("best_win_ratio", self.best_win_ratio)
        self.pure_mcts_playout_num = checkpoint.get("pure_mcts_playout_num", self.pure_mcts_playout_num)
        self.lr_multiplier = checkpoint.get("lr_multiplier", self.lr_multiplier)
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
        self._log("训练开始")
        try:
            for i in range(self.game_batch_num):
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
            if self.stop_requested:
                self._log(f"训练已软停止: {self.stop_reason}")
            # 确保日志落盘，避免中断导致 TensorBoard 数据丢失
            self.writer.flush()
            self.writer.close()


if __name__ == '__main__':
    pipeline = TrainPipeline()
    pipeline.run()
