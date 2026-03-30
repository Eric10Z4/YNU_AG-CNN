"""
优化器：SGD / Adam + L2 正则化
"""
import torch


class Optimizer:
    def __init__(self, params, lr=0.001, weight_decay=0.0):
        self.params = params
        self.lr = lr
        self.weight_decay = weight_decay
        self.state = {}
    
    def step(self):
        raise NotImplementedError
    
    def zero_grad(self):
        for p in self.params.values():
            if p is not None and p.grad is not None:
                p.grad.zero_()
    
    def _add_weight_decay(self, grad, name):
        if self.weight_decay > 0 and 'W' in name:
            return grad + self.weight_decay * self.params[name]
        return grad


class SGD(Optimizer):
    """随机梯度下降 + 动量"""
    def __init__(self, params, lr=0.001, weight_decay=0.0, momentum=0.9):
        super().__init__(params, lr, weight_decay)
        self.momentum = momentum
        for name in params:
            self.state[name] = torch.zeros_like(params[name])
    
    def step(self):
        for name, param in self.params.items():
            if param is None or param.grad is None:
                continue
            grad = self._add_weight_decay(param.grad, name)
            self.state[name] = self.momentum * self.state[name] + grad
            param.data -= self.lr * self.state[name]


class Adam(Optimizer):
    """自适应学习率"""
    def __init__(self, params, lr=0.001, weight_decay=0.0, 
                 beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(params, lr, weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        for name in params:
            self.state[name] = {
                'm': torch.zeros_like(params[name]),
                'v': torch.zeros_like(params[name])
            }
    
    def step(self):
        self.t += 1
        for name, param in self.params.items():
            if param is None or param.grad is None:
                continue
            grad = self._add_weight_decay(param.grad, name)
            m, v = self.state[name]['m'], self.state[name]['v']
            
            m.data = self.beta1 * m + (1 - self.beta1) * grad
            v.data = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
            
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            
            param.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
