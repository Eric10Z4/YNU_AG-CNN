# cpp_mcts_alphaZero.py
import numpy as np
import copy

def softmax(x):
    """计算 softmax 概率分布，减去最大值以防止数值溢出。"""
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode(object):
    """MCTS 树节点，记录状态统计信息。"""
    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}       # 子节点字典 {action: TreeNode}
        self._n_visits = 0        # 节点访问次数 N(s, a)
        self._Q = 0               # 动作的平均价值 Q(s, a)
        self._u = 0               # PUCT 探索附加值 U(s, a)
        self._P = prior_p         # 神经网络输出的先验概率 P(s, a)

    def expand(self, action_priors):
        """使用神经网络输出的 (合法动作, 概率) 列表展开当前叶子节点。"""
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """在子节点中选择 PUCT 得分最高的一个。返回: (action, 选中的 TreeNode)"""
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def get_value(self, c_puct):
        """计算并返回 PUCT 节点价值：Q(s, a) + U(s, a)"""
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def update(self, leaf_value):
        """更新当前节点的访问量和平均胜率 (Q值)。"""
        self._n_visits += 1
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """反向传播：将叶子节点价值一路向上更新至根节点。注意视角切换，价值需取反。"""
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def is_leaf(self):
        """检查是否为未展开的叶子节点"""
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """蒙特卡洛树搜索核心调度器。"""
    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """
        policy_value_fn: 接收 board，返回 (action_probs, leaf_value) 的网络前向函数。
        c_puct: 探索常数。
        n_playout: 每次落子前的 MCTS 模拟次数。
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        """执行单次 MCTS 模拟推演。"""
        node = self._root
        
        # 1. 选择 (Selection)
        while not node.is_leaf():
            action, node = node.select(self._c_puct)
            state.do_move(action)

        # 2. 评估 (Evaluation): 呼叫神经网络
        action_probs, leaf_value = self._policy(state)
        legal_set = set(state.availables)
        legal_action_probs = [(a, p) for a, p in action_probs if a in legal_set]
        if not legal_action_probs and len(state.availables) > 0:
            uniform_p = 1.0 / len(state.availables)
            legal_action_probs = [(a, uniform_p) for a in state.availables]

        # 3. 扩展 (Expansion) 或 结算真实胜负
        end, winner = state.game_end()
        if not end:
            node.expand(legal_action_probs)
        else:
            if winner == -1:  # 平局
                leaf_value = 0.0
            else:
                leaf_value = (1.0 if winner == state.current_player else -1.0)

        # 4. 回溯 (Backup)
        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp=1e-3):
        """
        执行 n_playout 次模拟，基于子节点访问次数计算并返回当前盘面所有合法动作及其概率。
        temp: 温度参数，控制输出策略的探索程度。
        """
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move):
        """复用搜索树：落子后将根节点指向对应的子节点，丢弃其他分支，节省算力。"""
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)


class MCTSPlayer(object):
    """AI 玩家实体，封装 MCTS 接口以接入游戏循环。"""
    def __init__(self, policy_value_function, c_puct=5, n_playout=2000, is_selfplay=0):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay
        self.player = None 

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def update_with_move(self, last_move):
        self.mcts.update_with_move(last_move)

    def get_action(self, board, temp=1e-3, return_prob=0):
        """根据当前棋盘状态返回执行动作。"""
        sensible_moves = board.availables
        move_probs = np.zeros(board.width * board.height)
        
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(board, temp)
            legal_set = set(sensible_moves)
            legal_pairs = [(a, p) for a, p in zip(acts, probs) if a in legal_set]
            if not legal_pairs:
                # 搜索树可能与当前棋盘不同步，重置后基于当前盘面重算一次
                self.mcts.update_with_move(-1)
                acts, probs = self.mcts.get_move_probs(board, temp)
                legal_pairs = [(a, p) for a, p in zip(acts, probs) if a in legal_set]
                if not legal_pairs:
                    raise ValueError(
                        f"MCTS 未返回任何合法动作: availables={len(sensible_moves)}, "
                        f"tree_children={len(self.mcts._root._children)}"
                    )

            acts = np.array([a for a, _ in legal_pairs], dtype=np.int64)
            probs = np.array([p for _, p in legal_pairs], dtype=np.float64)
            probs = probs / np.sum(probs)
            move_probs[acts.tolist()] = probs
            
            if self._is_selfplay:
                # 训练模式：增加 Dirichlet 噪声以提升探索多样性
                p = 0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs)))
                move = np.random.choice(acts, p=p)
                self.mcts.update_with_move(move)
            else:
                # 实战模式：直接按照概率分布选择动作
                move = np.random.choice(acts, p=probs)
                self.mcts.update_with_move(move)
                
            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("警告：棋盘已满")
            return -1
