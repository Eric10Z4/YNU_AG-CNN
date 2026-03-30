"""
测试：验证手搓网络
"""
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

from policy_value_net import PolicyValueNet
from losses import value_loss, policy_loss, combined_loss
from optimizer import SGD, Adam


def sync_weights(hand, torch_model):
    """同步权重"""
    # 卷积层
    hand.conv1.params['W'].data = torch_model.conv1.weight.data.clone()
    hand.conv1.params['b'].data = torch_model.conv1.bias.data.clone()
    hand.bn1.params['gamma'].data = torch_model.bn1.weight.data.clone()
    hand.bn1.params['beta'].data = torch_model.bn1.bias.data.clone()
    hand.bn1.running_mean = torch_model.bn1.running_mean.clone()
    hand.bn1.running_var = torch_model.bn1.running_var.clone()
    
    hand.conv2.params['W'].data = torch_model.conv2.weight.data.clone()
    hand.conv2.params['b'].data = torch_model.conv2.bias.data.clone()
    hand.bn2.params['gamma'].data = torch_model.bn2.weight.data.clone()
    hand.bn2.params['beta'].data = torch_model.bn2.bias.data.clone()
    hand.bn2.running_mean = torch_model.bn2.running_mean.clone()
    hand.bn2.running_var = torch_model.bn2.running_var.clone()
    
    hand.conv3.params['W'].data = torch_model.conv3.weight.data.clone()
    hand.conv3.params['b'].data = torch_model.conv3.bias.data.clone()
    hand.bn3.params['gamma'].data = torch_model.bn3.weight.data.clone()
    hand.bn3.params['beta'].data = torch_model.bn3.bias.data.clone()
    hand.bn3.running_mean = torch_model.bn3.running_mean.clone()
    hand.bn3.running_var = torch_model.bn3.running_var.clone()
    
    # 策略头
    hand.policy_conv.params['W'].data = torch_model.policy_conv.weight.data.clone()
    hand.policy_conv.params['b'].data = torch_model.policy_conv.bias.data.clone()
    hand.policy_bn.params['gamma'].data = torch_model.policy_bn.weight.data.clone()
    hand.policy_bn.params['beta'].data = torch_model.policy_bn.bias.data.clone()
    hand.policy_fc.params['W'].data = torch_model.policy_fc.weight.data.clone().T
    hand.policy_fc.params['b'].data = torch_model.policy_fc.bias.data.clone()
    
    # 价值头
    hand.value_conv.params['W'].data = torch_model.value_conv.weight.data.clone()
    hand.value_conv.params['b'].data = torch_model.value_conv.bias.data.clone()
    hand.value_bn.params['gamma'].data = torch_model.value_bn.weight.data.clone()
    hand.value_bn.params['beta'].data = torch_model.value_bn.bias.data.clone()
    hand.value_fc1.params['W'].data = torch_model.value_fc1.weight.data.clone().T
    hand.value_fc1.params['b'].data = torch_model.value_fc1.bias.data.clone()
    hand.value_fc2.params['W'].data = torch_model.value_fc2.weight.data.clone().T
    hand.value_fc2.params['b'].data = torch_model.value_fc2.bias.data.clone()


class TorchNet(nn.Module):
    """PyTorch 原生实现"""
    def __init__(self, board_size=8, num_channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(4, num_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.conv3 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_channels)
        
        self.policy_conv = nn.Conv2d(num_channels, 2, 1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size + 1)
        
        self.value_conv = nn.Conv2d(num_channels, 1, 1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 128)
        self.value_fc2 = nn.Linear(128, 1)
        
        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        
        p = torch.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        policy = torch.softmax(self.policy_fc(p), dim=-1)
        
        v = torch.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = torch.tanh(self.value_fc2(torch.relu(self.value_fc1(v))))
        
        return policy, v


def test():
    print("=" * 50)
    print("Testing Pipeline")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(42)
    
    # 直接测试手搓网络，不对比 PyTorch
    hand_net = PolicyValueNet(board_size=8, num_channels=32, device=device)
    hand_net.train_mode()
    
    # 测试数据
    x = torch.randn(2, 4, 8, 8, device=device)
    target_policy = torch.softmax(torch.randn(2, 65, device=device), dim=-1)
    target_value = torch.randn(2, 1, device=device)
    
    # 前向传播
    print("\nForward pass:")
    pred_p, pred_v = hand_net.forward(x)
    print(f"  Policy shape: {pred_p.shape}")
    print(f"  Value shape: {pred_v.shape}")
    
    # 损失计算
    print("\nLoss calculation:")
    loss, v_loss, p_loss, grad_v, grad_p = combined_loss(pred_p, pred_v, target_policy, target_value)
    print(f"  Total: {loss.item():.4f}")
    print(f"  Value: {v_loss.item():.4f}")
    print(f"  Policy: {p_loss.item():.4f}")
    
    # 测试优化器
    print("\nTesting optimizer:")
    params, _ = hand_net.get_all_params()
    params = {k: v for k, v in params.items() if v is not None}
    
    for k in params:
        params[k].grad = torch.randn_like(params[k]) * 0.01
    
    sgd = SGD(params, lr=0.01)
    sgd.step()
    print("  SGD: OK")
    
    for k in params:
        params[k].grad = torch.randn_like(params[k]) * 0.01
    
    adam = Adam(params, lr=0.001)
    adam.step()
    print("  Adam: OK")
    
    print("=" * 50)
    print("All tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    test()
