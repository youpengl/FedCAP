import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
# plt.rcParams['font.sans-serif']=['Times New Roman']
# plt.rcParams['axes.unicode_minus']=False
fig = plt.figure(figsize=(20, 5))
width = 0.25
bar_loc = [-width*3, -width*2, -width, 0, width, width*2, width*3]
ind = np.arange(0, 3*2, 2)
color = ['#F2F2F2', '#D5D5D5', '#A1A1A1', '#4F4F4F', '#E2E8F0', '#FADBD8']
edgecolor = ['black', 'black', 'black', 'black', '#3282b8', '#810000']
hatch = [None, None, None, None, '\\\\\\', '---']

# load data
atkrs = [0.3, 0.3, 0.3, 0.0]
jrs = [1.0, 0.2, 1.0]
ncs = [20, 100, 36]
sufs = [[['_lamda0.1', '_lamda0.01', '_lamda0.01', '_lamda1.0'],
        ['_RAMmean', '_RAMmean', '_RAMmean', '_RAMmean'],
        ['_RAMmean', '_RAMmean', '_RAMmean', '_RAMmean'],
        ['_lamda0.1_alpha10_phi0.3_normT10', '_lamda10.0_alpha10_phi0.3_normT10', '_lamda1.0_alpha10_phi0.1_normT10', '_lamda1.0_alpha10_phi0.3_normT10'],
        ['_O10', '_O10', '_O10', '_O10'],
        ['', '', '', ''],
        ['', '', '', '']],
        
        [['_lamda0.1', '_lamda0.01', '_lamda0.1', '_lamda0.1'],
        ['_RAMmean', '_RAMmean', '_RAMmean', '_RAMmean'],
        ['_RAMmean', '_RAMmean', '_RAMmean', '_RAMmean'],
        ['_lamda0.1_alpha10_phi0.2_normT10', '_lamda0.1_alpha10_phi0.1_normT10', '_lamda0.1_alpha10_phi0.3_normT10', '_lamda0.5_alpha10_phi0.1_normT10'],
        ['_O10', '_O10', '_O10', '_O10'],
        ['', '', '', ''],
        ['', '', '', '']],
        
        [['_lamda0.1', '_lamda0.01', '_lamda0.01', '_lamda0.1'],
        ['_RAMmean', '_RAMmean', '_RAMmean', '_RAMmean'],
        ['_RAMmean', '_RAMmean', '_RAMmean', '_RAMmean'],
        ['_lamda0.1_alpha4_phi0.2_normT10', '_lamda0.1_alpha2_phi0.1_normT10', '_lamda0.1_alpha2_phi0.2_normT10', '_lamda0.1_alpha2_phi0.1_normT10'],
        ['_O18', '_O18', '_O18', '_O18'],
        ['', '', '', ''],
        ['', '', '', '']],
        ]
INFO = {}
for idx, d in enumerate(['cifar10_pat', 'emnist_group', 'wisdm_nature']):
    INFO[d] = {}
    for jdx, alg in enumerate(['Ditto', 'FedAvg', 'FedAvg-FT', 'FedCAP', 'FedFomo', 'FedRoD', 'Local']):
        results = []
        for kdx, atk in enumerate(['A1', 'A3', 'A4', 'B']):
            if alg == 'Local' and atk != 'B':
                continue
            atkr = atkrs[kdx]
            jr = jrs[idx]
            nc = ncs[idx]
            suf = sufs[idx][jdx][kdx]
            if alg == 'FedAvg-FT':
                alg_ = 'FedAvg'
            elif alg == 'FedRoD':
                alg_ = 'FedROD'
            else:
                alg_ = alg
            file = "../results/npz/{}_{}_{}_{}_bz10_lr0.01_gr100_ep5_jr{}_nc{}_seed0{}.npz".format(d, alg_, atk, atkr, jr, nc, suf)
            with np.load(file, allow_pickle=True) as f:
                if alg != 'Local':
                    test_acc_g = f['test_acc_g'][-1]*100
                test_acc_p = f['test_acc_p'][-1]*100
                # test_acc_g = np.random.uniform(low=0, high=1, size=101)[-1]*100
                # test_acc_p = np.random.uniform(low=0, high=1, size=101)[-1]*100
                max_test_acc = max(test_acc_g, test_acc_p)
            if alg == 'FedAvg':
                results.append(test_acc_g)
            elif alg in ['FedAvg-FT', 'Local']:
                results.append(test_acc_p)
            else:
                results.append(max_test_acc)
                
        if alg != 'Local':
            results_ = [results[-1], results[0], results[2], results[1]]
        INFO[d][alg] = results_
        
for i, atk in enumerate(['Benign', 'LF', 'SF', 'MR']):

    if i == 0:
        width = 0.25
    else: 
        width = 0.25
        ind = np.arange(0, 3*2, 2)
    ax = plt.subplot(2, 2, i+1)
    if i == 0:
        ind = np.arange(0, 2.3*3, 2.3)
        x = ind - width * 4
        y = [INFO['cifar10_pat']['Local'][0], INFO['emnist_group']['Local'][0], INFO['wisdm_nature']['Local'][0]]
        axl = ax.bar(x, y, width, color='white', label='Local',
        hatch=None, edgecolor='black')
        ax.bar_label(axl, fmt='%.1f', fontsize=10, rotation=50)
        ax.set_xticks(ind)
        
    for j, method in enumerate(['FedAvg', 'FedAvg-FT', 'Ditto', 'FedRoD', 'FedFomo', 'FedCAP']):
        x = ind + bar_loc[j]
        y = [INFO['cifar10_pat'][method][i], INFO['emnist_group'][method][i], INFO['wisdm_nature'][method][i]]
        axl = ax.bar(x, y, width, color=color[j], label=method,
        hatch=hatch[j], edgecolor=edgecolor[j])
        ax.bar_label(axl, fmt='%.1f', fontsize=10, rotation=50)
        ax.set_xticks(ind)

    ax.set_ylim((0, 100))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.set_xticklabels(('CIFAR-10', 'EMNIST', 'WISDM'))
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0,
                    chartBox.width*1.1,
                    chartBox.height * 0.85])
    
    if i == 0 or i == 2:
        ax.set_ylabel('Test Accuracy', fontsize=18)
    ax.set_title('{}'.format(atk), fontsize=18)
    ax.tick_params(labelsize=14)

lines, labels = fig.axes[0].get_legend_handles_labels()
fig.legend(lines, labels, ncol=8, bbox_to_anchor=(0.8, 1), fontsize=14)
# fig.tight_layout()
# plt.plot()
plt.savefig('Figure7.pdf', bbox_inches='tight', pad_inches=0.0)
