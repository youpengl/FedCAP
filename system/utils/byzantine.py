import numpy as np
from copy import deepcopy
import torch
from utils import tools

# Random
def A2(k=None, v=None, malicious_cid=None, scaling_factor=None):
    if k != None:
        flatten_ki = torch.cat([kk.reshape((-1, 1)) for kk in k], dim=0) 
        std = torch.std(flatten_ki, unbiased=False).item()
        k = [torch.tensor(np.random.normal(0, std, size=kk.shape).astype('float32')).to(flatten_ki.device) for kk in k]
        return k
    else:
        for i in malicious_cid:
            flatten_vi = torch.cat([vv.reshape((-1, 1)) for vv in v[i]], dim=0) 
            std = torch.std(flatten_vi, unbiased=False).item()
            v[i] = [torch.tensor(np.random.normal(0, std, size=vv.shape).astype('float32')).to(flatten_vi.device) for vv in v[i]]
        return v


# Model Replacement
def A3(k=None, v=None, malicious_cid=None, scaling_factor=None):

    if k != None:
        k = [kk * scaling_factor for kk in k]
        return k
    
    else:
        scaling_factor = len(v) 
        for i in malicious_cid:
            v[i] = [vv * scaling_factor for vv in v[i]]
        return v

# Sign Flipping
def A4(k=None, v=None, malicious_cid=None, scaling_factor=None):
    if k != None:
        k = [-kk for kk in k]
        return k
    else:
        for i in malicious_cid:
            v[i] = [-vv for vv in v[i]]  
        return v
    
# LIE
def A5(k=None, v=None, malicious_cid=None, scaling_factor=None):

    num_byzantine = len(malicious_cid)
    num_uploaded_clients = len(v)
    s = torch.floor_divide(num_uploaded_clients, 2) + 1 - num_byzantine
    cdf_value = (num_uploaded_clients - num_byzantine - s) / (num_uploaded_clients - num_byzantine)
    dist = torch.distributions.normal.Normal(torch.tensor(0.0), torch.tensor(1.0))
    z_max = dist.icdf(cdf_value)
    num_layers = len(v[0])
    aggregated_param = []

    for i in range(num_layers):
        a = []
        for vv in v:
            for j, param in enumerate(vv):
                if j == i:
                    a.append(param)
                    break
        
        aggregated_v = torch.stack(a)
        mu = torch.mean(aggregated_v, dim=0)
        sigma = torch.std(aggregated_v, dim=0)

        result = mu - sigma * z_max

        aggregated_param.append(result)

    if k != None:
        return aggregated_param
    
    else:
        for i in malicious_cid:
            v[i] = aggregated_param
            
        return v
    
# Min-Max
def A6(k=None, v=None, malicious_cid=None, scaling_factor=None):


    flat_v = []
    for vv in v:
        flat_v.append(torch.cat([torch.reshape(vvv, (-1,)) for vvv in vv]))
    aggregated_v = torch.stack(flat_v)
    agg_v_mu = torch.mean(aggregated_v, dim=0)
    agg_v_std = -1.0 * torch.std(aggregated_v, dim=0)
    # search for optimal gamma
    gamma_init = 2
    e = 0.02
    step, gamma = gamma_init/2, gamma_init
    gamma_succ = 0
    while abs(gamma_succ-gamma) > e:
        grad_m = agg_v_mu + gamma * agg_v_std
        max_dist_m = tools.max_distance(aggregated_v, grad_m)
        max_dist_b = tools.max_pairwise_distance(aggregated_v)
        if  max_dist_m <= max_dist_b :
            gamma_succ = gamma
            gamma += step/2
        else:
            gamma -= step/2
        step = max(step/2, 0.1)
    
    result = agg_v_mu + gamma_succ * agg_v_std
    mal_v = []
    start_idx = 0
    for i, param in enumerate(vv):
        param_=result[start_idx:start_idx+len(param.view(-1))].reshape(param.data.shape)
        start_idx=start_idx+len(param.data.view(-1))
        mal_v.append(param_)

    if k != None:
        return mal_v

    else:
        for i in malicious_cid:
            v[i] = mal_v

        return v

# Min-Sum
def A7(k=None, v=None, malicious_cid=None, scaling_factor=None):

    flat_v = []
    for vv in v:
        flat_v.append(torch.cat([torch.reshape(vvv, (-1,)) for vvv in vv]))
    aggregated_v = torch.stack(flat_v)
    agg_v_mu = torch.mean(aggregated_v, dim=0)
    agg_v_std = -1.0 * torch.std(aggregated_v, dim=0)
    # search for optimal gamma
    gamma_init = 2
    e = 0.02
    step, gamma = gamma_init/2, gamma_init
    gamma_succ = 0
    while abs(gamma_succ-gamma) > e:
        grad_m = agg_v_mu + gamma * agg_v_std
        max_dist_m = tools.sum_distance(aggregated_v, grad_m)
        max_dist_b = tools.max_sum_distance(aggregated_v)
        if  max_dist_m <= max_dist_b :
            gamma_succ = gamma
            gamma += step/2
        else:
            gamma -= step/2
        step = max(step/2, 0.1)
    
    result = agg_v_mu + gamma_succ * agg_v_std
    mal_v = []
    start_idx = 0
    for i, param in enumerate(vv):
        param_=result[start_idx:start_idx+len(param.view(-1))].reshape(param.data.shape)
        start_idx=start_idx+len(param.data.view(-1))
        mal_v.append(param_)

    if k != None:
        return mal_v
    
    else:
        for i in malicious_cid:
            v[i] = mal_v       
        return v

# IPM
def A8(k=None, v=None, malicious_cid=None, scaling_factor=None):

    if scaling_factor == None:
        scaling_factor = len(v)
        
    benign_v = []
    for i in range(len(v)):
        if i not in malicious_cid:
            benign_v.append(v[i])

    num_layers = len(benign_v[0])
    aggregated_param = []

    for i in range(num_layers):
        a = []
        for vv in benign_v:
            for j, param in enumerate(vv):
                if j == i:
                    a.append(param)
                    break
        
        aggregated_v = torch.stack(a)
        mu = torch.mean(aggregated_v, dim=0)
        epsilon = scaling_factor
        result = -1.0 * epsilon * mu
        
        aggregated_param.append(result)

    if k != None:
        return aggregated_param
    
    else:
        for i in malicious_cid:
            v[i] = aggregated_param
        return v