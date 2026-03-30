import random
import numpy as np
from collections import deque
import torch
import os
import sys
from collections import defaultdict

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
        self.model_dir = os.path.join(ROOT_DIR, "models")
        self.current_model_path = os.path.join(self.model_dir, "current_policy.pth")
        self.best_model_path = os.path.join(self.model_dir, "best_policy.pth")
        os.makedirs(self.model_dir, exist_ok=True)

        # 网络与优化器
        self.policy_value_net = PolicyValueNet(self.board_width, num_channels=64, device=self.device)
        params, _ = self.policy_value_net.get_all_params()
        self.optimizer = Adam(params, lr=self.learn_rate)
        
        # MCTS 玩家
        self.mcts_player = MCTSPlayer(self.policy_value_fn, self.c_puct, self.n_playout, is_selfplay=1)
        self.episode_len = 0

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
            
            while True:
                move, move_probs = self.mcts_player.get_action(self.board, temp=self.temp, return_prob=1)
                states.append(self.board.current_state())
                mcts_probs.append(move_probs)
                current_players.append(self.board.current_player)
                
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
                    break

    def policy_update(self):
        # 采样与网络权重更新
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
        
        for i in range(self.epochs):
            self.policy_value_net.zero_grad()
            act_probs, value = self.policy_value_net.forward(state_batch)

            # 将 225 维 MCTS 目标补齐到 226 维（pass 动作监督为 0）
            target_policy_full = torch.zeros_like(act_probs)
            target_policy_full[:, :-1] = mcts_probs_batch

            # 手算 Loss 和 梯度（与网络 226 维输出对齐）
            total_loss, loss_v, loss_p, grad_v, grad_p = combined_loss(
                act_probs, value, target_policy_full, winner_batch
            )

            # 手动反向传播与优化
            self.policy_value_net.backward(grad_p, grad_v)
            self.optimizer.step()

            with torch.no_grad():
                new_probs, _ = self.policy_value_net.forward(state_batch)
                kl = torch.mean(
                    torch.sum(
                        old_probs * (torch.log(old_probs + 1e-10) - torch.log(new_probs + 1e-10)),
                        dim=1
                    )
                ).item()
            if kl > self.kl_targ * 4:
                break

        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5
        self.optimizer.lr = self.learn_rate * self.lr_multiplier

        explained_var_old = 1 - torch.var(winner_batch - old_v) / (torch.var(winner_batch) + 1e-10)
        explained_var_new = 1 - torch.var(winner_batch - value.detach()) / (torch.var(winner_batch) + 1e-10)
        print(
            f"kl:{kl:.5f}, lr_mult:{self.lr_multiplier:.3f}, "
            f"loss:{total_loss.item():.4f}, v_loss:{loss_v.item():.4f}, p_loss:{loss_p.item():.4f}, "
            f"ev_old:{explained_var_old.item():.3f}, ev_new:{explained_var_new.item():.3f}"
        )
             
        return total_loss.item(), loss_v.item(), loss_p.item()
    
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
        print(
            f"num_playouts:{self.pure_mcts_playout_num}, "
            f"win:{win_cnt[1]}, lose:{win_cnt[2]}, tie:{win_cnt[-1]}"
        )
        return win_ratio

    def _capture_model_state(self):
        params, _ = self.policy_value_net.get_all_params()
        return {k: v.detach().cpu().clone() for k, v in params.items() if v is not None}

    def _save_model(self, path):
        checkpoint = {
            "model_state": self._capture_model_state(),
            "best_win_ratio": self.best_win_ratio,
            "pure_mcts_playout_num": self.pure_mcts_playout_num,
            "lr_multiplier": self.lr_multiplier,
        }
        torch.save(checkpoint, path)

    def run(self):
        # 主循环
        print("训练开始")
        try:
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                print(f"Batch {i+1}: 游戏步数 {self.episode_len}, Buffer容量 {len(self.data_buffer)}")
                
                if len(self.data_buffer) >= self.batch_size:
                    loss, loss_v, loss_p = self.policy_update()
                    print(f"更新完毕 | Loss: {loss:.4f} (V: {loss_v:.4f}, P: {loss_p:.4f})")
                
                if (i + 1) % self.check_freq == 0:
                    print(f"current self-play batch: {i+1}")
                    win_ratio = self.policy_evaluate()
                    self._save_model(self.current_model_path)
                    if win_ratio > self.best_win_ratio:
                        print("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        self._save_model(self.best_model_path)
                        if self.best_win_ratio == 1.0 and self.pure_mcts_playout_num < 5000:
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
                    
        except KeyboardInterrupt:
            print("\n训练终止")

if __name__ == '__main__':
    pipeline = TrainPipeline()
    pipeline.run()
