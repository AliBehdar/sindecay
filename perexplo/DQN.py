import hydra
from omegaconf import DictConfig
from utility.utility import seed
from policy.train_test import Model_TrainTest

@hydra.main(version_base="1.1", config_path="./conf", config_name="config")      
def main(cfg: DictConfig):
    seed(cfg)  
    DRL = Model_TrainTest(cfg) 

    if cfg.train:
        DRL.train()
    else:
        DRL.test(max_episodes = cfg.max_episodes)
if __name__ == '__main__':   
    
    main()