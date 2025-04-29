from pathlib import Path
import hydra
from omegaconf import DictConfig
from datasets import *
from models import *
from helper import *

@hydra.main(config_path="../config", config_name="config.yaml")
def train_test_classifier(cfg: DictConfig) -> None:

    source_dir = Path(__file__).resolve().parent
    cfg = update_abs_path(cfg, source_dir)
    
    for model_type in cfg.modelTypes:
        print(f"Starting {model_type} model")
       
        train_loader, test_loader_cln, _, test_loader_trg = data_loader(cfg, model_type)
        net, pre_trained = get_conv_net(cfg, model_type, pre_trained=True) 
        
        if pre_trained == False:
            net = train_conv_net(cfg, net, model_type, train_loader, test_loader_cln, test_loader_trg)
        
        print("Clean Results:")
        results_cln = test_conv_net(cfg, net, model_type, test_loader_cln)
        print(results_cln)

        ### Only while testing the BadNet/Backdoored model
        # print("\n\nPoisoned Results:")
        # results_trg = test_conv_net(cfg, net, model_type, test_loader_trg)
        # print(results_trg)
        
if __name__ == '__main__':
    train_test_classifier()