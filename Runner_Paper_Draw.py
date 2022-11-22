import matplotlib.pyplot as plt
import numpy as np
from alisuretool.Tools import Tools


def abl_k():
    labels = ['5', '7', '9', '11', '13']
    acc = [94.91, 95.13, 96.68, 96.20, 96.33]

    x = np.arange(len(labels))  # the label locations
    width = 0.5  # the width of the bars

    fig, ax = plt.subplots(figsize=(4,3))
    rects1 = ax.bar(x, acc, width, label='AUC')
    # line1 = ax.plot(x, acc, color="blue", linestyle='--', marker='o', linewidth=2, markersize=8)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel(r'$K$')
    ax.set_ylabel('AUC')
    # ax.set_title('Scores by group and gender')
    ax.set_xticks(x, labels)
    # ax.legend()
    ax.set_ylim(92, 98)
    ax.bar_label(rects1, padding=4)

    fig.tight_layout()

    plt.show()
    fig.savefig(Tools.new_dir("./result/figure/Abl_K.pdf"))
    pass


def abl_c():
    labels = ['20', '30', '40', '50', '60']
    acc = [93.74, 96.35, 96.68, 96.15, 94.71]

    x = np.arange(len(labels))  # the label locations
    width = 0.5  # the width of the bars

    fig, ax = plt.subplots(figsize=(4, 3))
    rects1 = ax.bar(x, acc, width, label='AUC')
    # line1 = ax.plot(x, acc, color="blue", linestyle='--', marker='o', linewidth=2, markersize=8)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel(r'$c_{num}$')
    ax.set_ylabel('AUC')
    # ax.set_title('Scores by group and gender')
    ax.set_xticks(x, labels)
    # ax.legend()
    ax.set_ylim(92, 98)
    ax.bar_label(rects1, padding=4)

    fig.tight_layout()

    plt.show()
    fig.savefig(Tools.new_dir("./result/figure/Abl_C.pdf"))
    pass


def abl_l():
    labels = ['2', '4', '6', '8']
    acc = [95.86, 96.21, 96.68, 95.16]

    x = np.arange(len(labels))  # the label locations
    width = 0.5  # the width of the bars

    fig, ax = plt.subplots(figsize=(4, 3))
    rects1 = ax.bar(x, acc, width, label='AUC')
    # line1 = ax.plot(x, acc, color="blue", linestyle='--', marker='o', linewidth=2, markersize=8)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel(r'$The\ number\ of\ GNN\ layers$')
    ax.set_ylabel('AUC')
    # ax.set_title('Scores by group and gender')
    ax.set_xticks(x, labels)
    # ax.legend()
    ax.set_ylim(92, 98)
    ax.bar_label(rects1, padding=4)

    fig.tight_layout()

    plt.show()
    fig.savefig(Tools.new_dir("./result/figure/Abl_L.pdf"))
    pass


if __name__ == '__main__':
    abl_k()
    abl_c()
    abl_l()
    pass