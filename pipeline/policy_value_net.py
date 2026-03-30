"""
双头网络：策略头 + 价值头
AlphaZero 风格
"""
import torch
import torch.nn as nn
import sys
import os

# 添加 core 模块路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

from conv import Conv2D
from batchnorm import BatchNorm2D
from activations import ReLU, Tanh
from Flatten import Flatten
from linear import Linear


class PolicyValueNet(nn.Module):
    """双头网络：Policy 预测落子，Value 预测胜率"""
    
    def __init__(self, board_size=8, num_channels=128, device='cuda'):
        super().__init__()
        self.board_size = board_size
        self.device = device
        
        # 公共特征提取层
        self.conv1 = Conv2D(4, num_channels, 3, padding=1, device=device)
        self.bn1 = BatchNorm2D(num_channels, device=device)
        
        self.conv2 = Conv2D(num_channels, num_channels, 3, padding=1, device=device)
        self.bn2 = BatchNorm2D(num_channels, device=device)
        
        self.conv3 = Conv2D(num_channels, num_channels, 3, padding=1, device=device)
        self.bn3 = BatchNorm2D(num_channels, device=device)
        
        # 策略头：输出落子概率 (1x1卷积，padding=0)
        self.policy_conv = Conv2D(num_channels, 2, kernel_size=1, padding=0, device=device)
        self.policy_bn = BatchNorm2D(2, device=device)
        self.policy_flatten = Flatten(device=device)
        self.policy_fc = Linear(2 * board_size * board_size, 
                                board_size * board_size + 1, device=device)
        
        # 价值头：输出局面价值 (1x1卷积，padding=0)
        self.value_conv = Conv2D(num_channels, 1, kernel_size=1, padding=0, device=device)
        self.value_bn = BatchNorm2D(1, device=device)
        self.value_flatten = Flatten(device=device)
        self.value_fc1 = Linear(board_size * board_size, 128, device=device)
        self.value_fc2 = Linear(128, 1, device=device)

        # 激活层（复用实例，便于后续反向传播缓存）
        self.relu1 = ReLU(device=device)
        self.relu2 = ReLU(device=device)
        self.relu3 = ReLU(device=device)
        self.relu_policy = ReLU(device=device)
        self.relu_value = ReLU(device=device)
        self.tanh_value = Tanh(device=device)
        
    def forward(self, x):
        # 公共特征提取
        x = self.relu1.forward(self.bn1.forward(self.conv1.forward(x)))
        x = self.relu2.forward(self.bn2.forward(self.conv2.forward(x)))
        x = self.relu3.forward(self.bn3.forward(self.conv3.forward(x)))
        
        # 策略头
        p = self.relu_policy.forward(self.policy_bn.forward(self.policy_conv.forward(x)))
        p = self.policy_flatten.forward(p)
        policy = torch.softmax(self.policy_fc.forward(p), dim=-1)
        
        # 价值头
        v = self.relu_value.forward(self.value_bn.forward(self.value_conv.forward(x)))
        v = self.value_flatten.forward(v)
        v = self.value_fc1.forward(v)
        value = self.tanh_value.forward(self.value_fc2.forward(v))
        
        return policy, value
    
    def train_mode(self):
        """切换到训练模式"""
        for bn in [self.bn1, self.bn2, self.bn3, self.policy_bn, self.value_bn]:
            bn.is_training = True
    
    def eval_mode(self):
        """切换到推理模式"""
        for bn in [self.bn1, self.bn2, self.bn3, self.policy_bn, self.value_bn]:
            bn.is_training = False
    
    def get_all_params(self):
        """获取所有参数"""
        params = {}
        grads = {}
        
        layers = {
            'conv1': self.conv1, 'bn1': self.bn1,
            'conv2': self.conv2, 'bn2': self.bn2,
            'conv3': self.conv3, 'bn3': self.bn3,
            'pc': self.policy_conv, 'pbn': self.policy_bn, 'pf': self.policy_fc,
            'vc': self.value_conv, 'vbn': self.value_bn, 
            'vf1': self.value_fc1, 'vf2': self.value_fc2
        }
        
        for name, layer in layers.items():
            for key in layer.params:
                params[f'{name}_{key}'] = layer.params[key]
                grads[f'{name}_{key}'] = layer.grads.get(key)
        
        return params, grads
    
    def zero_grad(self):
        """清空所有梯度"""
        for layer in [self.conv1, self.bn1, self.conv2, self.bn2, self.conv3, self.bn3,
                      self.policy_conv, self.policy_bn, self.policy_fc,
                      self.value_conv, self.value_bn, self.value_fc1, self.value_fc2]:
            for key in layer.grads:
                if layer.grads[key] is not None:
                    layer.grads[key].zero_()
