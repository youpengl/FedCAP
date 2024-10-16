import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
fig = plt.figure(figsize=(10, 4))

linestyles = ['--', '-.', 'dotted', (5, (10, 3)), '-']
atks = ["LIE", "Min-Max", "Min-Sum", "IPM"]
methods = ['Ditto', "Ditto+ClusteredFL", 'Ditto+Median', 'Ditto+Trim.', "FedCAP"]
DATASETS = ['CIFAR-10', "EMNIST"]
colors = ["#CF4B3E", "#F19839", "#4689BD", "#59AA4B", "#A080C4"]
import pandas as pd

sufs = [[['_lamda1.0', '_lamda0.1_RAMtrim', '_lamda0.5_RAMcluster', '_lamda0.5_RAMmedian', '_lamda0.1_alpha10_phi0.2_normT10'],
        ['_lamda0.01', '_lamda0.1_RAMtrim', '_lamda1.0_RAMcluster', '_lamda0.1_RAMmedian', '_lamda0.1_alpha10_phi0.3_normT10'],
        ['_lamda0.1', '_lamda0.1_RAMtrim', '_lamda0.1_RAMcluster', '_lamda0.1_RAMmedian', '_lamda0.1_alpha10_phi0.3_normT10'],
        ['_lamda0.01', '_lamda0.1_RAMtrim', '_lamda0.1_RAMcluster', '_lamda0.1_RAMmedian', '_lamda0.1_alpha10_phi0.3_normT10']],
        
        [['_lamda0.01', '_lamda0.1_RAMtrim', '_lamda0.1_RAMcluster', '_lamda0.1_RAMmedian', '_lamda0.5_alpha10_phi0.1_normT10'],
        ['_lamda0.1', '_lamda0.1_RAMtrim', '_lamda0.1_RAMcluster', '_lamda0.1_RAMmedian', '_lamda0.5_alpha10_phi0.3_normT10'],
        ['_lamda0.1', '_lamda0.1_RAMtrim', '_lamda0.1_RAMcluster', '_lamda0.1_RAMmedian', '_lamda0.5_alpha10_phi0.1_normT10'],
        ['_lamda0.01', '_lamda0.1_RAMtrim', '_lamda0.5_RAMcluster', '_lamda0.1_RAMmedian', '_lamda0.5_alpha10_phi0.1_normT10']]]

INFO = {}
for idx, d in enumerate(['cifar10_pat', 'emnist_group']):
    INFO[d] = {}
    for jdx, atk in enumerate(['A5', 'A6', 'A7', 'A8']):
        INFO[d][atk] = {}
        for kdx, alg in enumerate(['mean', 'cluster', 'median', 'trim', 'FedCAP']):
            if alg == 'mean':
                alg_ = 'Ditto'
            elif alg == 'FedCAP':
                alg_ = alg
            else:
                alg_ = 'DittoAGRs'
            if d == 'cifar10_pat':
                jr = 1.0
                nc = 20
            elif d == 'emnist_group':
                jr = 0.2
                nc = 100
            suf = sufs[idx][jdx][kdx]
            file = "../results/npz/{}_{}_{}_0.3_bz10_lr0.01_gr100_ep5_jr{}_nc{}_seed0{}.npz".format(d, alg_, atk, jr, nc, suf)
            with np.load(file, allow_pickle=True) as f:
                test_acc_g = f['test_acc_g']
                test_acc_p = f['test_acc_p']
                # test_acc_g = np.random.uniform(low=0, high=1, size=101)
                # test_acc_p = np.random.uniform(low=0, high=1, size=101)
                max_test_accs = test_acc_p[1:] if test_acc_g[-1] < test_acc_p[-1] else test_acc_g[1:]
            INFO[d][atk][alg] = max_test_accs
count = 1
for d_idx, dataset in enumerate(DATASETS):
    for i, atk in enumerate(atks):
        ax1 = plt.subplot(2, 4, count)
        count+=1
        # csv = pd.read_csv('Figure8/Ditto_AGR_{}.csv'.format(dataset))
        for j, alg in enumerate(methods):
            # y = np.array(csv.iloc[:, 5*i+(j+1)])*100
            y = np.array(INFO[['cifar10_pat', 'emnist_group'][d_idx]][['A5', 'A6', 'A7', 'A8'][i]][['mean', 'cluster', 'median', 'trim', 'FedCAP'][j]])*100
            # print(y[-1])
            x = y.shape[0]
            ax1.plot(np.arange(x), y, label='{}'.format(alg), linestyle=linestyles[j], linewidth=2, color=colors[j])
        if d_idx == 0:
            ax1.set_title('{}'.format(atks[i]), fontsize=16)
        if i == 0:
            ax1.set_ylabel('{}'.format(DATASETS[d_idx]), fontsize=16, rotation='vertical')
        ax1.tick_params(labelsize=14)
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(50))
        if count < 5:
            ax1.set_yticks((50, 81, 10))
            ax1.set_ylim(50, 90)
        else:
            ax1.set_yticks((0, 81, 20))
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(20))
        chartBox = ax1.get_position()
        if d_idx == 0:
            ax1.set_position([chartBox.x0, chartBox.y0-0.07,
                            chartBox.width *0.95 ,
                            chartBox.height * 0.75])
            lines, labels = fig.axes[0].get_legend_handles_labels()
        else:
            ax1.set_position([chartBox.x0, chartBox.y0+0.01,
                            chartBox.width *0.95 ,
                            chartBox.height * 0.75])
            lines, labels = fig.axes[0].get_legend_handles_labels()

fig.legend(lines, labels, ncol=3, bbox_to_anchor=(0.88, 1), fontsize=16)
fig.text(0.03, 0.45, 'Test Accuracy', va='center', fontsize=16, rotation='vertical')
fig.text(0.5, 0.0, 'Global Rounds', ha='center', fontsize=16)
# fig.text(0.0, 0.5, 'Test Accuracy', va='center', rotation='vertical', fontsize=16)
# fig.tight_layout()
#plt.plot()
plt.savefig('Figure8.pdf', bbox_inches='tight', pad_inches=0.0)
