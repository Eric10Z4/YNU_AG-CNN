import torch
import math
from base import Layer
from im2col import im2col,col2im
import initializers

class Conv2D(Layer):
    #传入的参数是：输入的通道数，输出的通道数，卷积核大小，步长，填充的大小
    def __init__(self, in_channels,out_channels,kernel_size=3,stride=1,padding=1,device='cuda'):
        super().__init__(device=device)
        self.in_c=in_channels
        self.out_c=out_channels
        self.k=kernel_size
        self.s=stride
        self.p=padding
        #权重的大小：【输入的通道数，输出的通道数，卷积核大小】
        w_shape=(out_channels,in_channels,kernel_size,kernel_size)
        fan_in=in_channels*kernel_size*kernel_size
        #写入我的参数
        self.params['W']=initializers.kaiming_normal(shape=w_shape,fan_in=fan_in,device=self.device)
        self.params['b']=initializers.zeros(shape=(out_channels,),device=self.device)

    def forward(self, x):
        self.x_shape=x.shape
        N,C,H,W=x.shape
        out_h=(H+2*self.p-self.k)//self.s+1
        out_w=(W+2*self.p-self.k)//self.s+1
        #进行降维，把图片展开
        self.x_cols=im2col(x,self.k,self.p,self.s)
        #把4D卷积核W展开为2D的矩阵
        W_mat=self.params['W'].view(self.out_c,-1)
        out=torch.matmul(W_mat,self.x_cols)
        #把矩阵又变成4D图片
        out=out.view(N,self.out_c,out_h,out_w)
        out=out+self.params['b'].view(1,-1,1,1)

        return out
    
    def backward(self, grad_output):
        N=grad_output.shape[0]
        #把误差和权重转成2D矩阵
        grad_out_2d=grad_output.view(N,self.out_c,-1)
        W_mat=self.params['W'].view(self.out_c,-1)

        self.grads['b']=torch.sum(grad_output,dim=(0,2,3))
        grad_W_batch=torch.matmul(grad_out_2d,self.x_cols.transpose(1,2))
        self.grads['W']=torch.sum(grad_W_batch,dim=0).view(self.out_c,self.in_c,self.k,self.k)
        grad_cols=torch.matmul(W_mat.T,grad_out_2d)
        grad_input=col2im(grad_cols,self.x_shape,self.k,self.p,self.s)

        return grad_input