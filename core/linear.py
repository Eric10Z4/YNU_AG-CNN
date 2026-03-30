#全连接层
import torch
from base import Layer
import initializers

class Linear(Layer):
    def __init__(self, in_features,out_features,device='cuda'):
        super().__init__(device=device)
        #确定形状大小
        w_shape=(in_features,out_features)
        b_shape=(out_features,)

        #使用Kaiming 初始化
        self.params['W']=initializers.kaiming_normal(
            shape=w_shape,
            fan_in=in_features,
            device=self.device
        )
        self.params['b'] = initializers.zeros(shape=b_shape, device=self.device)

    def forward(self,x):
        self.x=x
        return torch.matmul(x,self.params['W'])+self.params['b']
        
    def backward(self,grad_output):
        #原来的Y就是现在的X
        grad_input=torch.matmul(grad_output,self.params['W'].T)
        #求得偏倒数，链式法则
        self.grads['W'] = torch.matmul(self.x.T, grad_output)
        self.grads['b'] = torch.sum(grad_output, dim=0)
        return grad_input