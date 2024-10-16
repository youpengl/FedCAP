import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
# plt.rcParams['font.sans-serif']=['Times New Roman']
# plt.rcParams['axes.unicode_minus']=False
fig = plt.figure(figsize=(8, 2.5))

x = ['0.1', '0.2', '0.3']

atkrs = [0.0, 0.3]
jrs = [1.0, 0.2, 1.0]
ncs = [20, 100, 36]
sufs = [[['_lamda1.0_alpha10_phi0.1_normT10', '_lamda1.0_alpha10_phi0.2_normT10', '_lamda1.0_alpha10_phi0.3_normT10'],
        ['_lamda0.1_alpha10_phi0.1_normT10', '_lamda0.1_alpha10_phi0.2_normT10', '_lamda0.1_alpha10_phi0.3_normT10']],
        
        [['_lamda0.5_alpha10_phi0.1_normT10', '_lamda0.5_alpha10_phi0.2_normT10', '_lamda0.5_alpha10_phi0.3_normT10'],
        ['_lamda0.5_alpha10_phi0.1_normT10', '_lamda0.5_alpha10_phi0.2_normT10', '_lamda0.5_alpha10_phi0.3_normT10']],
        
        [['_lamda0.1_alpha2_phi0.1_normT10', '_lamda0.1_alpha2_phi0.2_normT10', '_lamda0.1_alpha2_phi0.3_normT10'],
        ['_lamda0.1_alpha2_phi0.1_normT10', '_lamda0.1_alpha2_phi0.2_normT10', '_lamda0.1_alpha2_phi0.3_normT10']]
        ]

plot_range = [(83, 86), (85, 87), (93, 95)]
data = {}
for idx, d in enumerate(['cifar10_pat', 'emnist_group', 'wisdm_nature']):
    data[['CIFAR-10', 'EMNIST', 'WISDM'][idx]] = []
    for jdx, atk in enumerate(['B', 'A8']):
        data[['CIFAR-10', 'EMNIST', 'WISDM'][idx]].append([])
        for kdx in range(len(sufs[idx][jdx])):
            atkr = atkrs[jdx]
            jr = jrs[idx]
            nc = ncs[idx]
            suf = sufs[idx][jdx][kdx]
            file = "../results/npz/{}_FedCAP_{}_{}_bz10_lr0.01_gr100_ep5_jr{}_nc{}_seed0{}.npz".format(d, atk, atkr, jr, nc, suf)
            with np.load(file, allow_pickle=True) as f:
                test_acc_g = f['test_acc_g'][-1]*100
                test_acc_p = f['test_acc_p'][-1]*100
                # test_acc_g = np.random.uniform(low=0, high=1, size=101)[-1]*100
                # test_acc_p = np.random.uniform(low=0, high=1, size=101)[-1]*100
                max_test_accs = max(test_acc_g, test_acc_p)
            data[['CIFAR-10', 'EMNIST', 'WISDM'][idx]][jdx].append(max_test_accs)
            
    data[['CIFAR-10', 'EMNIST', 'WISDM'][idx]].append(plot_range[idx])

# print(data)
dataset = list(data.keys())

for i in range(len(dataset)):
    ax1 = plt.subplot(1, 3, i+1)
    ax1.plot(x, data[dataset[i]][0], label='Benign', marker='s', linewidth=3, markersize=10)
    ax1.plot(x, data[dataset[i]][1], label='IPM', marker='^', linewidth=3, markersize=10)
    ax1.set_title('{}'.format(dataset[i]), fontsize=18)
# ax1.set_ylabel('Test Accuracy', fontsize=18)
    ax1.tick_params(labelsize=14)
    # ax1.set_xticks(x, rotation=0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(x, rotation=0)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(4))
    ax1.set(ylim=data[dataset[i]][2], yticks=np.arange(data[dataset[i]][2][0], data[dataset[i]][2][1]+1, 1), xticks=x)
    ax1.set_ylim()
    if i == 0:
        ax1.set_ylabel('Test Accuracy', fontsize=16)
    elif i == 1:
        chartBox = ax1.get_position()
        ax1.set_xlabel(r'$\phi$', fontsize=16)
        ax1.set_position([chartBox.x0, chartBox.y0,
                    chartBox.width,
                    chartBox.height* 1.2])
lines, labels = fig.axes[0].get_legend_handles_labels()
fig.legend(lines, labels, ncol=4, bbox_to_anchor=(0.73, 1), fontsize=16)
fig.tight_layout()
# plt.show()
plt.savefig('Figure11.pdf', bbox_inches='tight', pad_inches=0.0)