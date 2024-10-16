# Complete code to generate the plot with the specified adjustments

import matplotlib.pyplot as plt
import numpy as np
# plt.rcParams['font.sans-serif']=['Times New Roman']
# plt.rcParams['axes.unicode_minus']=False
fractions = np.array([10, 20, 30, 40])
methods = ["M-Krum", "Median", "RFA", "Trim.", "ClusteredFL", "FLTrust", "Ditto", "FedCAP"]
methods_abb = ["MKrum", "Median", "RFA", "Trim", "Cluster", "FLTrust", "Ditto", "FedCAP"]
attacks = ["LIE", "Min-Max", "Min-Sum", "IPM"]
markers = ['s', 'o', 'X', '^']
algs = ["FedAvg", "FedAvg", "FedAvg", "FedAvg", "FedAvg", "FLTrust", "Ditto", "FedCAP"]
jrs = [1.0, 0.2]
ncs = [20, 100]
sufs = [
        # cifar10
        [[['_RAMmkrum']*4]*4,
        [['_RAMmedian']*4]*4,
        [['_RAMrfa']*4]*4,
        [['_RAMtrim']*4]*4,
        [['_RAMcluster']*4]*4,
        [['']*4]*4,
        
        [['_lamda1.0', '_lamda1.0', '_lamda1.0', '_lamda0.1'],
         ['_lamda0.1', '_lamda1.0', '_lamda0.01', '_lamda0.1'],
         ['_lamda1.0', '_lamda0.5', '_lamda0.1', '_lamda0.1'],
         ['_lamda0.01', '_lamda0.1', '_lamda0.01', '_lamda0.1']],
        
        [['_lamda0.1_alpha10_phi0.3_normT10', '_lamda1.0_alpha10_phi0.1_normT10', '_lamda0.1_alpha10_phi0.2_normT10', '_lamda0.1_alpha10_phi0.2_normT10'],
         ['_lamda0.1_alpha10_phi0.3_normT10', '_lamda0.1_alpha10_phi0.2_normT10', '_lamda0.1_alpha10_phi0.3_normT10', '_lamda0.1_alpha10_phi0.3_normT10'],
         ['_lamda0.1_alpha10_phi0.3_normT10', '_lamda0.1_alpha10_phi0.1_normT10', '_lamda0.1_alpha10_phi0.3_normT10', '_lamda0.5_alpha10_phi0.3_normT10'],
         ['_lamda0.1_alpha10_phi0.3_normT10', '_lamda0.1_alpha10_phi0.2_normT10', '_lamda0.1_alpha10_phi0.3_normT10', '_lamda0.5_alpha10_phi0.1_normT10']]],
        
        # emnist
        [[['_RAMmkrum']*4]*4,
        [['_RAMmedian']*4]*4,
        [['_RAMrfa']*4]*4,
        [['_RAMtrim']*4]*4,
        [['_RAMcluster']*4]*4,
        [['']*4]*4,
        
        [['_lamda0.1', '_lamda0.5', '_lamda0.01', '_lamda0.1'],
        ['_lamda0.1', '_lamda0.1', '_lamda0.1', '_lamda0.1'],
        ['_lamda0.1', '_lamda0.1', '_lamda0.1', '_lamda0.1'],
        ['_lamda0.01', '_lamda0.01', '_lamda0.01', '_lamda0.1']],
        
        [['_lamda0.5_alpha10_phi0.1_normT10', '_lamda0.5_alpha10_phi0.1_normT10', '_lamda0.5_alpha10_phi0.1_normT10', '_lamda0.5_alpha10_phi0.2_normT10'],
        ['_lamda0.5_alpha10_phi0.2_normT10', '_lamda0.5_alpha10_phi0.1_normT10', '_lamda0.5_alpha10_phi0.3_normT10', '_lamda0.5_alpha10_phi0.1_normT10'],
        ['_lamda1.0_alpha10_phi0.1_normT10', '_lamda1.0_alpha10_phi0.1_normT10', '_lamda0.5_alpha10_phi0.1_normT10', '_lamda0.5_alpha10_phi0.1_normT10'],
        ['_lamda0.5_alpha10_phi0.2_normT10', '_lamda0.5_alpha10_phi0.1_normT10', '_lamda0.5_alpha10_phi0.1_normT10', '_lamda0.5_alpha10_phi0.1_normT10']], 
    ]]
        
INFO = {}   
for idx, d in enumerate(['cifar10_pat', 'emnist_group']):
    INFO[d] = {}
    for jdx, alg in enumerate(methods_abb):
        INFO[d][alg] = []
        for kdx, atk in enumerate(['A5', 'A6', 'A7', 'A8']):
            INFO[d][alg].append([])
            for ldx, atkr in enumerate([0.1, 0.2, 0.3, 0.4]):
                alg_ = algs[jdx]
                jr = jrs[idx]
                nc = ncs[idx]
                suf = sufs[idx][jdx][kdx][ldx]
                file = "../results/npz/{}_{}_{}_{}_bz10_lr0.01_gr100_ep5_jr{}_nc{}_seed0{}.npz".format(d, alg_, atk, atkr, jr, nc, suf)
                with np.load(file, allow_pickle=True) as f:
                    test_acc_g = f['test_acc_g'][-1]*100
                    test_acc_p = f['test_acc_p'][-1]*100
                    # test_acc_g = np.random.uniform(low=0, high=1, size=101)[-1]*100
                    # test_acc_p = np.random.uniform(low=0, high=1, size=101)[-1]*100
                    max_test_accs = max(test_acc_g, test_acc_p)
                INFO[d][alg][kdx].append(max_test_accs)
            

# Function to plot data without a legend
def plot_method_no_legend(i, ax, method, data, fractions):
    # linestyles = ['-', '--', ':', '-.', (0, (3, 1, 1, 1)), (0, (3, 5, 1, 5)), (0, (5, 10))]
    linestyles = ['-', '--', ':', '-.']
    for attack_idx, (attack, linestyle) in enumerate(zip(attacks, linestyles)):
        ax.plot(fractions, data[method][attack_idx], linestyle=linestyle, linewidth=3, label=attack, marker=markers[attack_idx], markersize=10)
    chartBox = ax.get_position()
    if i >= 8:
        ax.set_position([chartBox.x0-0.02, chartBox.y0-0.02,
                        chartBox.width * 0.8,
                        chartBox.height * 0.6])
    
    else:
        ax.set_position([chartBox.x0-0.02, chartBox.y0-0.1,
                        chartBox.width * 0.8,
                        chartBox.height * 0.6])
    
    ax.set_xticks(fractions)
    ax.set_ylim(0, 100)
    ax.tick_params(labelsize=18)

    ax.set_title(methods[methods_abb.index(method)], size=16)


# Create figure and subplots
fig, axs = plt.subplots(2, 8, figsize=(20, 5.5), constrained_layout=True)

# Plot data on subplots
for i, (ax, method) in enumerate(zip(axs.flatten(), methods_abb * 2)):
    if i < 8:
        plot_method_no_legend(i, ax, method, INFO['cifar10_pat'], fractions)
    else:
        plot_method_no_legend(i, ax, method, INFO['emnist_group'], fractions)

    if i % 7 != 0:
        ax.set_ylabel('')
    # if i == 0:
    #     # ax.yaxis.set_label_position('right')
    #     ax.set_ylabel('CIFAR-10', size=20)
    # elif i == 8:
    #     # ax.yaxis.set_label_position('right')
    #     ax.set_ylabel('EMNIST', size=20)

# Remove subplot x and y labels
# for ax in axs.flatten():
#     ax.label_outer()
# Set common titles (moved to the correct positions)
fig.text(0.48, 0, 'Fraction of Malicious Clients (%)', ha='center', fontsize=18)
fig.text(0.05, 0.4, 'Test Accuracy', va='center', fontsize=18, rotation='vertical')
fig.text(0.065, 0.2, 'CIFAR-10', va='center', fontsize=16, rotation='vertical')
fig.text(0.065, 0.55, 'EMNIST', va='center', fontsize=16, rotation='vertical')
# fig.subplots_adjust(hspace=0.1)
# Create a legend for the whole figure
# lines = [plt.Line2D([0], [0], color='black', linestyle=ls) for ls in linestyles]
# fig.legend(lines, labels, loc='upper center', ncol=len(attacks), bbox_to_anchor=(0.5, 1.02))
lines, labels = fig.axes[0].get_legend_handles_labels()
fig.legend(lines, labels, ncol=len(attacks), bbox_to_anchor=(0.65, 0.8), fontsize=16)
# fig.tight_layout()
#plt.show()
plt.savefig('Figure10.pdf', bbox_inches='tight', pad_inches=0.0)