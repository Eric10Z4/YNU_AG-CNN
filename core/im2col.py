'''
为啥我要引入这个呢 ，其实呢就是为了避免重复的循环，加入我有一个4*4 的图片
且我要使用一个3*3 的卷积核，步长为1,那么我就可以使用4*3*3 个小图，即4个长度为9的向量
然后拼接为一个9*4的二维大矩阵，然后卷积核变成1*9的这样就能直接完成一步的矩阵乘法
'''
import torch

def get_im2col_indices(x_shape,kernel_size,padding=1,stride=1,device='cuda'):#张量尺寸，卷积核大小，填充，布长
    N,C,H,W=x_shape#批量大小 通道数 高度 宽度
    out_h=(H+2*padding-kernel_size)//stride+1
    out_w=(W+2*padding-kernel_size)//stride+1 #下棋就不要填充了
    #计算卷积核内部的相对坐标
    #先生成大小为K的一维向量，然后重复K次，
    i0=torch.repeat_interleave(torch.arange(kernel_size, device=device), kernel_size)
    #然后将上述的行坐标数组重复C次
    i0 = torch.tile(i0, (C,))
    #然后沿着第一维度拼接，长度就可以得到为C*K*K
    j0 = torch.tile(torch.arange(kernel_size, device=device), (kernel_size * C,))
    #计算滑动窗口在整张大图上的锚点坐标偏移
    i1 = stride * torch.repeat_interleave(torch.arange(out_h, device=device), out_w)
    j1 = stride * torch.tile(torch.arange(out_w, device=device), (out_h,))

    #相对坐标 + 锚点偏移 = 绝对坐标
    i = i0.view(-1, 1) + i1.view(1, -1)
    j = j0.view(-1, 1) + j1.view(1, -1)
    
    #生成通道的索引坐标
    k = torch.repeat_interleave(torch.arange(C, device=device), kernel_size * kernel_size).view(-1, 1)

    return k, i, j


def im2col(x,kernel_size,padding=1,stride=1):
    p=padding
    N,C,H,W=x.shape
    #手动进行填充
    x_padded=torch.zeros((N, C, H + 2*p, W + 2 * p), dtype=x.dtype, device=x.device)
    if p>0:#填充的需要填内容
        x_padded[:,:,p:-p,p:-p]=x
    else:
        x_padded=x
    
    #拿到通道索引，行，列坐标
    k, i, j = get_im2col_indices(x.shape, kernel_size, padding, stride, device=x.device)
    # 直接根据坐标矩阵，把 4D 张量里的元素“抠”出来排成序列
    # cols 形状: [N,   C * kernel_size * kernel_size,    out_h * out_w]
    cols = x_padded[:, k, i, j] #具体的过程上面已经说明了
    return cols

def col2im(cols,x_shape,kernel_size=3,padding=1,stride=1):
    N,C,H,W=x_shape#批量大小 通道数 高度 宽度
    p=padding
    x_padded = torch.zeros((N, C, H + 2 * p, W + 2 * p), dtype=cols.dtype, device=cols.device)
    #拿到通道索引，行，列坐标
    k, i, j = get_im2col_indices(x_shape, kernel_size, padding, stride, device=cols.device)
    # 因为滑动窗口有重叠，同一个像素会被算好几次梯度，必须累加，就是链式法则形成的原因
    for b in range(N):
        # 每次只取第 b 个样本的梯度矩阵进行累加折叠
        x_padded[b].index_put_((k, i, j), cols[b], accumulate=True)
    if p > 0:#同理填充的需要删内容
        return x_padded[:, :, p:-p, p:-p]
    return x_padded




