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

def compute_gradient_divergence_batch(old_nets, global_model, party_list):
    """
    批量计算所有参与 client 的方向散度信号（深层 Cosine Divergence）。
    
    d_i = 1 - cos(drift_i, mean_drift)
    只用 layer4 + fc 的参数，降维提升区分度。
    
    理论支撑：
    - SCAFFOLD (Karimireddy et al., ICML 2020): 方向不一致是收敛变慢的根源
    - FedNova (Wang et al., NeurIPS 2020): 方向偏差导致 objective inconsistency
    
    参数：
        old_nets: dict, 上一轮各 client 的 local model（old_nets_pool[-1]）
        global_model: 当前轮聚合后的 global model
        party_list: list, 本轮参与的 client id 列表
    
    返回：
        list of float, 每个 client 的方向散度 d_i ∈ [0, 2]
    """
    import torch.nn.functional as F
    
    # 目标层：只用深层参数（task-specific layers）
    target_keywords = ['layer4', 'fc', 'linear', 'l3']
    
    # 提取 global model 深层参数
    global_params = []
    for name, p in global_model.named_parameters():
        if any(kw in name for kw in target_keywords):
            global_params.append(p.detach().cpu().view(-1))
    global_flat = torch.cat(global_params)
    
    # 提取每个 client 的 drift 向量
    drifts = []
    for cid in party_list:
        local_params = []
        for name, p in old_nets[cid].named_parameters():
            if any(kw in name for kw in target_keywords):
                local_params.append(p.detach().cpu().view(-1))
        local_flat = torch.cat(local_params)
        drifts.append(local_flat - global_flat)
    
    # 计算平均 drift 方向
    drift_stack = torch.stack(drifts)  # [N, dim]
    mean_drift = drift_stack.mean(dim=0)  # [dim]
    mean_norm = torch.norm(mean_drift)
    
    # 计算每个 client 与平均方向的 cosine divergence
    d_values = []
    for drift in drifts:
        drift_norm = torch.norm(drift)
        if mean_norm < 1e-10 or drift_norm < 1e-10:
            d_values.append(0.0)
        else:
            cos_sim = F.cosine_similarity(
                drift.unsqueeze(0), mean_drift.unsqueeze(0)
            ).item()
            cos_sim = max(-1.0, min(1.0, cos_sim))  # 数值稳定
            d_values.append(1.0 - cos_sim)
    
    return d_values

def compute_distillation_gain(local_model, global_model, dataloader, device, n_batches=3):
    """
    用 KL 散度度量 local 和 global 在该 client 数据上的输出分歧。
    
    g_i = KL(p_global || p_local)
    
    计算位置：client 端（local training 开始前）
    通信开销：上传 1 个 float（可忽略）
    计算开销：3 batch forward pass，约 0.05s
    
    语义：g_i 越大 → local 偏离 global 越严重 → 需要更强 contrastive 拉回
          恒为非负，不会出现 non-IID 下"全负"的问题
    """
    import torch.nn.functional as F
    
    local_model.eval()
    global_model.eval()
    
    kl_sum, n = 0.0, 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            if batch_idx >= n_batches:
                break
            data = data.to(device)
            
            _, _, local_out = local_model(data)
            _, _, global_out = global_model(data)
            
            # temperature=0.5，与 contrastive loss 保持一致
            local_log_prob = F.log_softmax(local_out / 0.5, dim=1)
            global_prob = F.softmax(global_out / 0.5, dim=1)
            
            kl = F.kl_div(local_log_prob, global_prob, reduction='batchmean')
            kl_sum += kl.item()
            n += 1
    
    return kl_sum / max(n, 1)

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