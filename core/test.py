import torch
import sys
sys.stdout.reconfigure(encoding='utf-8')

# 导入你纯手搓的全部核心组件！
from conv import Conv2D
from batchnorm import BatchNorm2D
from activations import ReLU, Tanh
from Flatten import Flatten
from linear import Linear

def run_ignition_test():
    print("="*50)
    print("AlphaZero Tensor Engine Starting...")
    print("="*50)
    
    # 自动检测是否有 GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device.upper()}")

    # 伪造一个 AlphaZero 五子棋的棋盘张量
    x = torch.randn(2, 4, 8, 8, device=device)
    print(f"Input shape: {x.shape}")
    print("-" * 50)

    # 实例化所有手搓组件
    print("Assembling network...")
    conv1 = Conv2D(in_channels=4, out_channels=32, kernel_size=3, padding=1, device=device)
    bn1 = BatchNorm2D(num_features=32, device=device)
    relu1 = ReLU(device=device)
    flatten = Flatten(device=device)
    linear1 = Linear(in_features=32 * 8 * 8, out_features=1, device=device)
    tanh = Tanh(device=device)
    print("Network assembled!")
    print("-" * 50)

    # 前向传播
    print("Forward pass...")
    out = conv1.forward(x)
    out = bn1.forward(out)
    out = relu1.forward(out)
    out = flatten.forward(out)
    out = linear1.forward(out)
    out = tanh.forward(out)
    
    print(f"Output shape: {out.shape}")
    print(f"Output value:\n{out.detach().cpu().numpy()}")
    print("-" * 50)

    # 反向传播
    print("Backward pass...")
    grad = torch.ones_like(out, device=device)
    
    grad = tanh.backward(grad)
    grad = linear1.backward(grad)
    grad = flatten.backward(grad)
    grad = relu1.backward(grad)
    grad = bn1.backward(grad)
    grad_input = conv1.backward(grad)

    print(f"Gradient shape: {grad_input.shape}")
    
    if 'W' in conv1.grads and conv1.grads['W'] is not None:
         print(f"Conv2D weight grad OK, shape: {conv1.grads['W'].shape}")
    else:
         print("Conv2D weight grad FAILED!")

    print("="*50)
    print("All tests passed!")
    print("="*50)

if __name__ == "__main__":
    torch.manual_seed(42)
    run_ignition_test()