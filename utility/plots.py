import matplotlib.pyplot as plt
import numpy as np
def plot(episode,cfg,reward_history,epsilon_history,loss_history):
        print('\n~~~~~~Interval Save: Model saved.\n')
        sma_reward = np.convolve(reward_history, np.ones(50)/50, mode='valid')
        max_reward=np.max(reward_history)
        min_reward=np.min(reward_history)
        #normalized_loss = np.interp(loss_history, (np.min(loss_history), np.max(loss_history)), (min_reward/2,max_reward))
        normalized_epsilon = np.interp(epsilon_history, (np.min(epsilon_history), np.max(epsilon_history)), (min_reward/4,max_reward))
        plt.plot(loss_history, label='Loss', color='#CB291A', alpha=0.8)
        
        plt.title("Training Progress")
        plt.xlabel("Episode")
        plt.ylabel("loss")
        plt.xlim(0, len(loss_history))
        plt.legend()
        plt.grid(True)

        if episode == cfg.max_episodes:
            plt.savefig('loss_progress.png', format='png', dpi=600, bbox_inches='tight')
            
        plt.tight_layout()
        #plt.show()
        #plt.clf()
        plt.close()

        plt.figure(figsize=(10, 6))
        
        #Plot Rewards,SMA 50 Reward ,Normalized Loss and Normalized Epsilon
        plt.plot(reward_history, label='Raw Reward', color='#F6CE3B', alpha=0.8)
        plt.plot(sma_reward, label='SMA 50 Reward', color='#385DAA')
        plt.plot(normalized_epsilon, label='Normalized Epsilon', color='green', alpha=0.8)
        
        #plt.plot(normalized_loss, label='Normalized Loss', color='#CB291A', alpha=0.8)
        
        plt.title("Training Progress")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True)

        # Save as file if last episode
        if episode == cfg.max_episodes:
            plt.savefig('training_progress.png', format='png', dpi=600, bbox_inches='tight')
            
        plt.tight_layout()
        #plt.show()
        #plt.clf()
        plt.close()
