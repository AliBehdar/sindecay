import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

def plot(episode, cfg, reward_history, epsilon_history, loss_history):
    print('\n~~~~~~Interval Save: Model saved.\n')

    plt.rcParams.update({
        'font.family': 'DejaVu Serif', 
        'font.size': 14,
        'axes.labelweight': 'bold',
        'axes.titlesize': 16,
        'axes.titleweight': 'bold',
        'axes.labelsize': 14,
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'lines.linewidth': 2.5,
        'grid.alpha': 0.4,
    })

    sma_reward = np.convolve(reward_history, np.ones(50)/50, mode='valid')

    plt.figure(figsize=(8, 8))
    plt.plot(loss_history, label='Loss', color='#CB291A', alpha=0.8)
    plt.title("Loss Progress", fontweight='bold')
    plt.xlabel("Episode", fontweight='bold')
    plt.ylabel("Loss", fontweight='bold')
    plt.xlim(0, len(loss_history))
    plt.legend()
    plt.grid(True, linestyle='--')

    if episode % 100 == 0:
        plt.savefig('loss_progress.png', format='png', dpi=600, bbox_inches='tight')
    plt.close()

    fig, ax1 = plt.subplots(figsize=(8, 8))

    ax1.plot(reward_history, label='Raw Reward', color='#F6CE3B', alpha=0.9)
    ax1.plot(sma_reward, label='SMA 50 Reward', color='#385DAA')
    ax1.set_xlabel("Episode", fontweight='bold')
    ax1.set_ylabel("Reward", fontweight='bold')
    ax1.grid(True, linestyle='--')

    if getattr(cfg, "env_name", None) == "LunarLander-v3":
        ax1.set_ylim(-600, 350)

        ax1.yaxis.set_major_locator(MultipleLocator(100))
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%d'))

        ax1.grid(which='major', linestyle='-', linewidth=0.8, alpha=0.6)
    if cfg.epsilon_greedy:
        ax2 = ax1.twinx()
        ax2.plot(epsilon_history, label='Epsilon', color='green', linestyle='-', alpha=0.8)
        ax2.set_ylabel("Epsilon", fontweight='bold')
        ax2.set_ylim(0, 2)
    else: 
        ax2 = ax1.twinx()
        ax2.plot(epsilon_history, label='Tempertuer', color='green', linestyle='-', alpha=0.8)
        ax2.set_ylabel("Tempertuer", fontweight='bold')
        ax2.set_ylim(0, 2)

    for label in ax1.get_yticklabels():
        label.set_fontweight('bold')
    for label in ax1.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax2.get_yticklabels():
        label.set_fontweight('bold')

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='best', frameon=True)

    plt.title("Training Progress", fontweight='bold')

    if episode % 100 == 0:
        plt.savefig('training_progress.png', format='png', dpi=600, bbox_inches='tight')

    plt.tight_layout()
    plt.close()
