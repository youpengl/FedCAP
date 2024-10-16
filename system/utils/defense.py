import torch
import numpy as np
from hdmedians import geomedian

class Register(dict):
    def __init__(self, *args, **kwargs):
        super(Register, self).__init__(*args, **kwargs)
        self._dict = {}

    def register(self, target):
        def add_item(key, value):
            if not callable(value):
                raise Exception(f"Error:{value} must be callable!")
            if key in self._dict:
                print(f"\033[31mWarning:\033[0m {value.__name__} already exists and will be overwritten!")
            self[key] = value
            return value

        if callable(target):
            return add_item(target.__name__, target)
        else:
            return lambda x : add_item(target, x)

    def __setitem__(self, key, value):
        self._dict[key] = value

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def __str__(self):
        return str(self._dict)

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def items(self):
        return self._dict.items()

agg_func = Register()

@agg_func.register('krum')
@torch.no_grad()
def krum(uploaded_updates, uploaded_weights, n_cl, n_sel, n_sel_byz):

    k = n_sel - n_sel_byz - 2
    assert k > 0
    uploaded_weights = [1.]
    # krum: return the parameter which has the lowest score defined as the sum of distance to its closest k vectors
    flattened_grads = []
    for i in range(len(uploaded_updates)):
        flattened_grads.append(torch.cat([param.reshape(-1, 1) for param in uploaded_updates[i]], dim=0).squeeze())
    distance = np.zeros((len(uploaded_updates), len(uploaded_updates)))
    for i in range(len(uploaded_updates)):
        for j in range(i+1, len(uploaded_updates)):
            distance[i][j] = torch.sum(torch.square(flattened_grads[i] - flattened_grads[j]))
            distance[j][i] = distance[i][j]

    score = np.zeros(len(uploaded_updates))
    for i in range(len(uploaded_updates)):
        score[i] = np.sum(np.sort(distance[i])[:k+1])  # the distance including itself, so k+1 not k

    selected_idx = np.argsort(score)[0]

    return [uploaded_updates[selected_idx]], uploaded_weights

@agg_func.register('mkrum')
@torch.no_grad()
def mkrum(uploaded_updates, uploaded_weights, n_cl, n_sel, n_sel_byz):

    m = n_sel - n_sel_byz
    k = m - 2
    assert k > 0

    flattened_grads = []
    for i in range(len(uploaded_updates)):
        flattened_grads.append(torch.cat([param.reshape(-1, 1) for param in uploaded_updates[i]], dim=0).squeeze())
    distance = np.zeros((len(uploaded_updates), len(uploaded_updates)))
    for i in range(len(uploaded_updates)):
        for j in range(i+1, len(uploaded_updates)):
            distance[i][j] = torch.sum(torch.square(flattened_grads[i] - flattened_grads[j]))
            distance[j][i] = distance[i][j]

    score = np.zeros(len(uploaded_updates))
    for i in range(len(uploaded_updates)):
        score[i] = np.sum(np.sort(distance[i])[:k+1])  # the distance including itself, so k+1 not k

    # multi-krum selects top-m 'good' vectors (defined by socre) (m=1: reduce to krum)
    selected_idx = np.argsort(score)[:m]

    uploaded_updates = [uploaded_updates[i] for i in range(len(uploaded_updates)) if i in selected_idx]
    uploaded_weights = [1/len(uploaded_updates) for i in range(len(uploaded_updates))]

    return uploaded_updates, uploaded_weights

@agg_func.register('median')
@torch.no_grad()
def median(uploaded_updates, uploaded_weights, n_cl, n_sel, n_sel_byz):

    uploaded_weights = [1.]
    if type(uploaded_updates) == torch.Tensor: # gas
        aggregated_param = uploaded_updates.median(dim=0)[0]
        device = uploaded_updates[0].device
    else:
        num_layers = len(uploaded_updates[0])
        aggregated_param = []
        device = uploaded_updates[0][0].device
        for i in range(num_layers):
            a = []
            for update in uploaded_updates:
                for j, param in enumerate(update):
                    if j == i:
                        a.append(param.clone().detach().cpu().numpy().flatten())
                        break
            aggregated_v = np.reshape(np.median(a, axis=0), newshape=param.shape)
            aggregated_param.append(torch.tensor(aggregated_v).to(device))

    return [aggregated_param], uploaded_weights

@agg_func.register('rfa')
@torch.no_grad()
def rfa(uploaded_updates, uploaded_weights, n_cl, n_sel, n_sel_byz):

    uploaded_weights = [1.]
    if type(uploaded_updates) == torch.Tensor: # gas
        device = uploaded_updates[0].device
        aggregated_param = torch.Tensor(geomedian(uploaded_updates.cpu().numpy(), axis=0)).to(device)
    else:
        num_layers = len(uploaded_updates[0])
        aggregated_param = []
        device = uploaded_updates[0][0].device
        for i in range(num_layers):
            a = []
            for update in uploaded_updates:
                for j, param in enumerate(update):
                    if j == i:
                        a.append(param.clone().detach().cpu().numpy().flatten())
                        break
            aggregated_v = np.reshape(geomedian(np.array(a), axis=0), newshape=param.shape)
            aggregated_param.append(torch.tensor(aggregated_v).to(device))

    return [aggregated_param], uploaded_weights

@agg_func.register('trim')
@torch.no_grad()
def trim(uploaded_updates, uploaded_weights, n_cl, n_sel, n_sel_byz):
    m = int(n_sel_byz / 2)
    assert n_sel - 2 * m > 0
    uploaded_weights = [1.]

    if type(uploaded_updates) == torch.Tensor: # gas
        a = torch.sort(uploaded_updates, dim=0)[0]
        a = a[m:len(uploaded_updates)-m, :]
        aggregated_param = a.mean(dim=0)
        device = uploaded_updates[0].device
    else:
        num_layers = len(uploaded_updates[0])
        aggregated_param = []
        device = uploaded_updates[0][0].device
        for i in range(num_layers):
            a = []
            for update in uploaded_updates:
                for j, param in enumerate(update):
                    if j == i:
                        a.append(param.clone().detach().cpu().numpy().flatten())
                        break

            a = np.array(a)
            a = np.sort(a, axis=0)
            a = a[m:len(uploaded_updates)-m, :]
            a = np.mean(a, axis=0)     
            a = np.reshape(a, newshape=param.shape)
            aggregated_param.append(torch.tensor(a).to(device))

    return [aggregated_param], uploaded_weights

@agg_func.register('cluster')
@torch.no_grad()
def cluster(uploaded_updates, uploaded_weights, n_cl, n_sel, n_sel_byz):
    num = len(uploaded_updates)
    flattened_grads = []
    for i in range(len(uploaded_updates)):
        flattened_grads.append(torch.cat([param.reshape(-1, 1) for param in uploaded_updates[i]], dim=0).squeeze())
    dis_max = np.zeros((num, num))
    for i in range(num):
        for j in range(i + 1, num):
            dis_max[i, j] = 1 - torch.nn.functional.cosine_similarity(
                flattened_grads[i], flattened_grads[j], dim=0
            )
            dis_max[j, i] = dis_max[i, j]
    dis_max[dis_max == -np.inf] = -1
    dis_max[dis_max == np.inf] = 1
    dis_max[np.isnan(dis_max)] = -1

    from sklearn.cluster import AgglomerativeClustering
    clustering = AgglomerativeClustering(
        affinity="precomputed", linkage="complete", n_clusters=2
    )
    clustering.fit(dis_max)
    
    flag = 1 if np.sum(clustering.labels_) > num // 2 else 0

    uploaded_updates = [uploaded_updates[i] for i in range(num) if i in np.where(clustering.labels_==flag)[0]]
    uploaded_weights = [uploaded_weights[i] for i in range(num) if i in np.where(clustering.labels_==flag)[0]]

    uploaded_weights = [uploaded_weights[i]/sum(uploaded_weights) for i in range(len(uploaded_weights))]

    return uploaded_updates, uploaded_weights