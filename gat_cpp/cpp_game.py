# cpp_game.py
import numpy as np

class Board(object):
    """
    五子棋的棋盘环境
    负责维护棋盘状态、执行落子动作、判定胜负以及生成当前盘面特征
    """
    def __init__(self, width=15, height=15, n_in_row=5):
        self.width = width
        self.height = height
        self.n_in_row = n_in_row
        # 玩家编号：1 和 2 (在对接神经网络时，通常会映射为 1 和 -1)
        self.players = [1, 2]
        
    def init_board(self, start_player=0):
        """初始化或重置棋盘状态"""
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception("棋盘尺寸不能小于连珠数要求！")
            
        self.current_player = self.players[start_player]  # 当前轮到的玩家
        self.availables = list(range(self.width * self.height))  # 所有合法的 1D 落子位置
        self.states = {}  # 记录棋盘状态。键: move(int), 值: player(int)
        self.last_move = -1  # 记录最后一步棋，用于加速胜负判定

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.width = self.width
        result.height = self.height
        result.n_in_row = self.n_in_row
        result.players = self.players[:]
        result.current_player = self.current_player
        result.availables = self.availables[:]
        result.states = dict(self.states)
        result.last_move = self.last_move
        return result

    def move_to_location(self, move):
        """降维: 1D -> 2D"""
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        """升维: 2D -> 1D"""
        if len(location) != 2:
            return -1
        h, w = location[0], location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        """
        生成 NCHW 标准的 4D 张量输入
        返回形状: [4, width, height]
        """
        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(self.states.keys())), np.array(list(self.states.values()))
            
            # 找到当前玩家和对手的落子位置
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            
            # 通道 0: 当前玩家的棋子
            square_state[0][move_curr // self.width, move_curr % self.width] = 1.0
            # 通道 1: 对手的棋子
            square_state[1][move_oppo // self.width, move_oppo % self.width] = 1.0
            # 通道 2: 最后落子的位置 (帮助 AI 聚焦局部战况)
            square_state[2][self.last_move // self.width, self.last_move % self.width] = 1.0
            
        # 通道 3: 当前玩家的颜色标志 (先手全 1.0，后手全 0.0)
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0  
            
        return square_state

    def do_move(self, move):
        """执行落子动作并切换玩家"""
        if move not in self.availables:
            raise ValueError(
                f"非法落子: move={move}, current_player={self.current_player}, "
                f"availables_size={len(self.availables)}, last_move={self.last_move}"
            )
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )
        self.last_move = move

    def has_a_winner(self):
        """向四个轴向发射射线检查胜利判定 (O(1) 极速判定)"""
        if len(self.states) < 5:
            return False, -1

        m = self.last_move
        p = self.states[m]
        h, w = m // self.width, m % self.width
        
        axes = [
            [(0, 1), (0, -1)],   
            [(1, 0), (-1, 0)],   
            [(1, 1), (-1, -1)],  
            [(1, -1), (-1, 1)]   
        ]

        for axis in axes:
            count = 1  
            for dir_h, dir_w in axis:
                for step in range(1, self.n_in_row):
                    next_h = h + step * dir_h
                    next_w = w + step * dir_w
                    if 0 <= next_h < self.height and 0 <= next_w < self.width:
                        next_m = next_h * self.width + next_w
                        if self.states.get(next_m) == p:
                            count += 1
                        else:
                            break 
                    else:
                        break 
            if count >= self.n_in_row:
                return True, p
                
        return False, -1

    def game_end(self):
        """检查游戏是否结束"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1


class Game(object):
    """
    控制游戏流程与界面渲染
    """
    def __init__(self, board):
        self.board = board

    def graphic(self, board, player1, player2):
        """在终端打印极简的 ASCII 棋盘"""
        width = board.width
        height = board.height

        print("\n" + "Player 1 with X".center(width * 3))
        print("Player 2 with O".center(width * 3) + "\n")
        
        print("  ", end="")
        for x in range(width):
            print(f"{x:2d}", end=" ")
        print()

        for h in range(height):
            print(f"{h:2d}", end=" ")
            for w in range(width):
                loc = h * width + w
                p = board.states.get(loc, -1)
                if p == player1:
                    print('X'.center(2), end=" ")
                elif p == player2:
                    print('O'.center(2), end=" ")
                else:
                    print('_'.center(2), end=" ")
            print()
        print()

    def start_play(self, player1, player2, start_player=0, is_shown=1):
        """
        🚀 引擎升级：现在它可以接收真正的 Player 对象了！
        无论是人类还是 AI，只要有 get_action 方法就能坐下来下棋。
        """
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) or 1 (player2 first)')
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        if hasattr(player1, "reset_player"):
            player1.reset_player()
        if hasattr(player2, "reset_player"):
            player2.reset_player()
        players = {p1: player1, p2: player2}
        
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)
            
        while True:
            current_player = self.board.current_player
            player_in_turn = players[current_player]
            if hasattr(player_in_turn, "update_with_move") and self.board.last_move != -1:
                player_in_turn.update_with_move(self.board.last_move)
            
            # 向玩家索要动作 (如果是人类就弹出 input，如果是 AI 就跑 MCTS)
            move = player_in_turn.get_action(self.board)
            self.board.do_move(move)
            
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
                
            is_end, winner = self.board.game_end()
            if is_end:
                if is_shown:
                    if winner != -1:
                        print(f"游戏结束！赢家是玩家 {winner}")
                    else:
                        print("游戏结束！平局！")
                return winner


class HumanPlayer(object):
    """人类玩家实体，负责把你的键盘输入转化为动作"""
    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        try:
            location = input(f"轮到你了 (玩家 {self.player})，请输入坐标 [行,列]: ")
            if isinstance(location, str):
                location = [int(n, 10) for n in location.split(",")]
            move = board.location_to_move(location)
        except Exception as e:
            move = -1
            
        if move == -1 or move not in board.availables:
            print("输入无效或该位置已有棋子，请重新输入！")
            move = self.get_action(board)
        return move

# === 测试入口 ===
if __name__ == '__main__':
    test_board = Board(width=15, height=15, n_in_row=5)
    game = Game(test_board)
    
    # 实例化两个人类玩家
    player1 = HumanPlayer()
    player2 = HumanPlayer()
    
    print("环境基建完成！测试：双人终端对战")
    # 让两个人类玩家进入对局
    game.start_play(player1, player2, start_player=0, is_shown=1)
