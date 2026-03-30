import torch
from base import Layer

class BatchNorm2D(Layer):
    #eps：可选参数，默认值为 1e-5，用于数值稳定性的小常数（防止除零）。
    # momentum：可选参数，默认值为 0.1，用于动量更新（如滑动平均）
    def __init__(self, num_features,eps=1e-5,momentum=0.1,device='cuda'):
        """
        将每一层的特征图强行拉回正态分布。
        """
        super().__init__(device=device)
        self.C=num_features
        self.eps=eps
        self.momentum=momentum
        self.is_training=True#用于进行训练模式的开关
        #缩放因子 gamma 和 平移因子 beta
        self.params['gamma']=torch.ones(self.C,device=self.device)
        self.params['beta']=torch.zeros(self.C,device=self.device)
        #全局统计量：推理(下棋)时用的均值和方差
        self.running_mean=torch.zeros(self.C,device=self.device)
        self.running_var=torch.ones(self.C,device=self.device)

    def forward(self, x):
        # x 形状: [N, C, H, W]
        # 测试模式
        if not self.is_training:
            mean=self.running_mean.view(1, self.C, 1, 1)
            var=self.running_var.view(1, self.C, 1, 1)
            x_hat=(x-mean)/torch.sqrt(var+self.eps)
            y=self.params['gamma'].view(1, self.C, 1, 1)*x_hat+self.params['beta'].view(1, self.C,1,1)
            return y
        # 训练模式
        #首先，我们需要把形状【1,C，1，1】变换，这样方便广播
        self.mean=torch.mean(x,dim=(0,2,3),keepdim=True)
        self.var=torch.var(x,dim=(0,2,3),unbiased=False,keepdim=True)#使用无偏估计法同时保持维度不变
        #核心操作：归一化，让其成为标准正态分布
        self.x_hat=(x-self.mean)/torch.sqrt(self.var+self.eps)
        #仿射变化用于正则化
        gamma=self.params['gamma'].view(1,self.C,1,1)
        beta=self.params['beta'].view(1,self.C,1,1)
        y=gamma*self.x_hat+beta
        #更新全局统计量，同时使用detach函数用来斩断梯度图，防止内存
        self.running_mean=(1-self.momentum)*self.running_mean+self.momentum*self.mean.squeeze().detach()
        self.running_var=(1-self.momentum)*self.running_var+self.momentum*self.var.squeeze().detach()

        return y
    
    def backward(self, grad_output):
        N, C, H, W = grad_output.shape
        D = N * H * W # 每个通道包含的像素点总数
        #计算沿着N，H，W三个维度的误差
        self.grads['gamma']=torch.sum(grad_output*self.x_hat,dim=(0,2,3))
        self.grads['beta']=torch.sum(grad_output,dim=(0,2,3))

        gamma=self.params['gamma'].view(1,C,1,1)
        std=torch.sqrt(self.var+self.eps)

        sum_dy=self.grads['beta'].view(1,C,1,1)
        sum_dy_x_hat=self.grads['gamma'].view(1,C,1,1)

        grad_input=(gamma/(D*std))*(D*grad_output-sum_dy-self.x_hat*sum_dy_x_hat)

        return grad_input


