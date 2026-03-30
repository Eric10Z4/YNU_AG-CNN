#我做了一个父亲模板这样方便后续的开发继承
import torch

class Layer:
    """""
    我们使用了CUDA,因为我们如果手搓向量加速很容易出现问题，从而在这个base的基础上，我们特以提醒。
    """""
    def __init__(self,device='cuda'):#默认用GPU
        # 存储该层所有可训练的参数 (比如 W 和 b)
        self.params={}
        # 存储损失函数对这些参数的梯度
        self.grads={}
        # 缓存前向传播时的输入张量
        self.x=None
        self.device=device
    
    def forward(self,x):
        raise NotImplementedError("子类必须实现 forward 方法！")
    
    def backward(self,grad_output):
        raise NotImplementedError("子类必须实现 backward 方法！")
