"""
损失函数：Value Loss + Policy Loss
"""
import torch


def value_loss(pred, target):
    """价值损失：均方误差"""
    diff = target - pred
    loss = torch.mean(diff ** 2)
    grad = -2.0 * diff / diff.numel()
    return loss, grad


def policy_loss(pred, target):
    """策略损失：交叉熵"""
    pred = torch.clamp(pred, min=1e-10)
    loss = -torch.sum(target * torch.log(pred)) / pred.size(0)
    grad = -target / pred / pred.size(0)
    return loss, grad


def combined_loss(pred_policy, pred_value, target_policy, target_value):
    """组合损失"""
    v_loss, grad_v = value_loss(pred_value, target_value)
    p_loss, grad_p = policy_loss(pred_policy, target_policy)
    total = v_loss + p_loss
    return total, v_loss, p_loss, grad_v, grad_p
