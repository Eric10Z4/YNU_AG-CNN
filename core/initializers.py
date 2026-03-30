import torch
import math

def kaiming_normal(shape,fan_in,device='cuda'):
    '''
    kaiming初始化是为了防止对于过深的神经网络层
    由于数据是完全的随机初始化，这样很容易出现导数NAN或者梯度消失
    核心的数学就是： 标准差=sqrt(2/fan_in)
    2是由于RELU砍去了一半
    '''
    std=math.sqrt(2.0/fan_in)

    return torch.randn(shape,device=device)*std

def zeros(shape,device='cuda'):
    '''
    初始化B
    '''
    return torch.zeros(shape,device=device)