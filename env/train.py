import random
import numpy as np
from collections import deque
import torch

from game import Board, Game
from mcts_alphaZero import MCTSPlayer

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
        self.learn_rate = 2e-3
        self.n_playout = 400
        self.c_puct = 5
        self.batch_size = 512
        self.buffer_size = 10000
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5
        self.check_freq = 50  # 每 50 次迭代评估一次模型
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 网络与优化器
        self.policy_value_net = PolicyValueNet(self.board_width, num_channels=128, device=self.device)
        params, _ = self.policy_value_net.get_all_params()
        self.optimizer = Adam(params, lr=self.learn_rate)
        
        # MCTS 玩家
        self.mcts_player = MCTSPlayer(self.policy_value_fn, self.c_puct, self.n_playout, is_selfplay=1)

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
        
        return zip(legal_positions, act_probs[legal_positions]), value

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

    def collect_selfplay_data(self):
        # 自对弈产数
        self.board.init_board()
        self.mcts_player.reset_player()
        states, mcts_probs, current_players = [], [], []
        
        while True:
            move, move_probs = self.mcts_player.get_action(self.board, temp=1.0, return_prob=1)
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
                self.data_buffer.extend(self.get_equi_data(play_data))
                return len(states)

    def policy_update(self):
        # 采样与网络权重更新
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = torch.tensor(np.array([d[0] for d in mini_batch]), dtype=torch.float32, device=self.device)
        mcts_probs_batch = torch.tensor(np.array([d[1] for d in mini_batch]), dtype=torch.float32, device=self.device)
        winner_batch = torch.tensor(np.array([d[2] for d in mini_batch]), dtype=torch.float32, device=self.device).unsqueeze(1)
        
        self.policy_value_net.train_mode()
        
        for i in range(self.epochs):
            self.policy_value_net.zero_grad()
            act_probs, value = self.policy_value_net.forward(state_batch)
            
            # 切除 Pass 动作对齐 225 维
            pred_policy = act_probs[:, :-1]
            
            # 手算 Loss 和 梯度
            total_loss, loss_v, loss_p, grad_v, grad_p = combined_loss(
                pred_policy, value, mcts_probs_batch, winner_batch
            )
            
            # 补齐 226 维回传梯度
            grad_p_padded = torch.zeros_like(act_probs)
            grad_p_padded[:, :-1] = grad_p
            
            # 手动反向传播与优化
            self.policy_value_net.backward(grad_p_padded, grad_v)
            self.optimizer.step()
            
        return total_loss.item(), loss_v.item(), loss_p.item()

    def policy_evaluate(self):
        # 评估模型：目前暂缺纯 MCTS 陪练，仅打印提示，后续可扩充
        print("\n--- 模型评估节点 ---")
        # 预留给人类或纯 MCTS 对打的接口
        pass

    def run(self):
        # 主循环
        print("训练开始")
        try:
            for i in range(1500):
                play_steps = self.collect_selfplay_data()
                print(f"Batch {i+1}: 游戏步数 {play_steps}, Buffer容量 {len(self.data_buffer)}")
                
                if len(self.data_buffer) > self.batch_size:
                    loss, loss_v, loss_p = self.policy_update()
                    print(f"更新完毕 | Loss: {loss:.4f} (V: {loss_v:.4f}, P: {loss_p:.4f})")
                
                if (i + 1) % self.check_freq == 0:
                    self.policy_evaluate()
                    
        except KeyboardInterrupt:
            print("\n训练终止")

if __name__ == '__main__':
    pipeline = TrainPipeline()
    pipeline.run()