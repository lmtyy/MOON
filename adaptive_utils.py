# adaptive_utils.py
"""
AdaMOON v2: Rank-Normalized Dual-Signal Adaptive Distillation
信号计算模块

设计原则：
1. gradient_divergence: server 端计算，零额外通信
2. distillation_gain: client 端计算，上传 1 个 float
3. rank_normalize: 保证区分度的数学保证
"""

import torch
import numpy as np
from scipy.stats import rankdata

def compute_gradient_divergence(local_model, global_model):
    """
    计算 local model 相对于 global model 的相对参数漂移。
    
    d_i = ||w_local - w_global|| / ||w_global||
    
    计算位置：server 端（收到 local model 后）
    通信开销：零（server 本来就有两个模型）
    计算开销：O(d)，约 0.01s
    
    理论支撑：FedProx (Li et al., MLSys 2020) Theorem 4
    """
    local_params = []
    global_params = []
    
    for lp, gp in zip(local_model.parameters(), global_model.parameters()):
        local_params.append(lp.detach().cpu().view(-1))
        global_params.append(gp.detach().cpu().view(-1))
    
    local_flat = torch.cat(local_params)
    global_flat = torch.cat(global_params)
    
    drift = local_flat - global_flat
    drift_norm = torch.norm(drift).item()
    global_norm = torch.norm(global_flat).item()
    
    d_i = drift_norm / (global_norm + 1e-8)
    return d_i

def compute_distillation_gain(local_model, global_model, dataloader, device, n_batches=3):
    """
    计算 global model 相对于 local model 的准确率优势。
    
    g_i = acc(global, D_i) - acc(local, D_i)
    
    计算位置：client 端（local training 开始前）
    通信开销：上传 1 个 float（可忽略）
    计算开销：3 batch forward pass，约 0.05s
    
    语义：g_i > 0 表示蒸馏对该 client 有价值
          g_i < 0 表示 local model 在自己数据上更强
    """
    local_model.eval()
    global_model.eval()
    
    local_correct = 0
    global_correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            if batch_idx >= n_batches:
                break
            data, target = data.to(device), target.to(device).long()
            
            # local model prediction
            _, _, out_local = local_model(data)
            local_correct += (out_local.argmax(1) == target).sum().item()
            
            # global model prediction
            _, _, out_global = global_model(data)
            global_correct += (out_global.argmax(1) == target).sum().item()
            
            total += target.size(0)
    
    local_acc = local_correct / max(total, 1)
    global_acc = global_correct / max(total, 1)
    
    g_i = global_acc - local_acc  # 允许负值
    return g_i

def rank_normalize(values):
    """
    将原始信号转为 [0, 1] 排名分数。
    
    数学性质：
    - 输出永远是均匀分布 U[0,1]，不管输入分布形状
    - n=20 时，λ_std ≈ (λ_max - λ_min) / √12 ≈ 0.013
    - 当所有值相同（IID 退化）→ 返回 0.5（不强行拉开差距）
    
    参数：
        values: list of float, 长度 = 参与本轮的 client 数
    返回：
        list of float, 每个值 ∈ [0, 1]
    """
    n = len(values)
    if n <= 1:
        return [0.5] * n
    
    values_arr = np.array(values, dtype=np.float64)
    
    # IID 退化保护：如果所有值几乎相同，返回均匀中心值
    if np.std(values_arr) < 1e-7:
        return [0.5] * n
    
    ranks = rankdata(values_arr, method='average')
    # 映射到 [0, 1]：rank 1 → 0, rank n → 1
    normalized = (ranks - 1.0) / (n - 1.0)
    return normalized.tolist()

def compute_adaptive_lambdas(d_values, g_values, lambda_min, lambda_max,
                              alpha_blend, lambda_emas, momentum, client_ids):
    """
    完整的 λ 计算流水线。
    
    参数：
        d_values: list, 本轮各 client 的梯度散度
        g_values: list, 本轮各 client 的蒸馏增益
        lambda_min: float, λ 下界 (0.005)
        lambda_max: float, λ 上界 (0.05)
        alpha_blend: float, d 和 g 的融合权重 (0.5)
        lambda_emas: dict, {client_id: ema_value}，会被原地更新
        momentum: float, EMA 动量 (0.9)
        client_ids: list, 本轮参与的 client id 列表
    
    返回：
        dict: {client_id: lambda_i}
    """
    n = len(d_values)
    
    # Step 1: Rank normalization
    d_rank = rank_normalize(d_values)
    g_rank = rank_normalize(g_values)
    
    # Step 2: 线性融合 + EMA
    lambda_dict = {}
    for idx in range(n):
        cid = client_ids[idx]
        s_i = alpha_blend * d_rank[idx] + (1 - alpha_blend) * g_rank[idx]
        raw_lambda = lambda_min + (lambda_max - lambda_min) * s_i
        
        # EMA 平滑
        if cid not in lambda_emas:
            lambda_emas[cid] = raw_lambda  # 首次不做 EMA
        else:
            lambda_emas[cid] = momentum * lambda_emas[cid] + (1 - momentum) * raw_lambda
        
        lambda_dict[cid] = lambda_emas[cid]
    
    return lambda_dict