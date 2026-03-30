#我们使用了ReLU函数作为激活函数
import torch
from base import Layer

class ReLU(Layer):
    def __init__(self,device='cuda'):
        super().__init__(device=device)
        #ReLU 不需要进行参数的更新
    
    def forward(self,x):
        self.x=x
        res=torch.maximum(x,torch.tensor(0.0,device=x.device))#一定要是在统一设备的标量
        return res
    
    def backward(self, grad_output): #FUCK JMY！！！
        flag=(self.x>0).float()
        grad_input=grad_output*flag
        return grad_input
    
class Tanh(Layer):
    """
    将胜负评放在 [-1, 1] 区间。
    """
    def __init__(self, device='cuda'):
        super().__init__(device=device)

    def forward(self, x):
        out=torch.tanh(x)
        self.out=out
        return out
    
    def backward(self, grad_output):
        grad_input=grad_output*(1.0-self.out**2)
        return grad_input
    
