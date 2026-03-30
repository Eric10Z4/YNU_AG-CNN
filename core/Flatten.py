import torch
from base import Layer

class Flatten(Layer):
    """
    管道转换配件：将 4D 张量拍扁成 2D 矩阵，专为 Linear 层供货。
    """

    def __init__(self, device='cuda'):
        super().__init__(device=device)
         
    def forward(self, x):
        #死死记住输入时的形状,从而方便我们回传
        self.x_shape=x.shape
        N=x.shape[0]
        out=x.view(N,-1)# 自动输出形状为【N，自动数】的向量
        return out

    def backward(self, grad_output):
        #拿到全连接层传回来的 2D 误差，像吹气球一样恢复成 4D 传给卷积层
        grad_input=grad_output.view(self.x_shape)
        return grad_input
