from pathlib import Path
import hydra
from omegaconf import DictConfig
from datasets import *
from models import *
from helper import *

@hydra.main(config_path="../config", config_name="config.yaml")
def train_test_autoencoder(cfg: DictConfig) -> None:

    source_dir = Path(__file__).resolve().parent
    cfg = update_abs_path(cfg, source_dir)
    train_loader, test_loader, _, _ = data_loader(cfg, model_type='autoencoder')

    encoder, decoder, pre_trained = get_autoencoder(cfg, pre_trained = True)
    if pre_trained == False:
        encoder, decoder = train_autoencoder(cfg, encoder, decoder, train_loader, test_loader)

if __name__ == '__main__':
    train_test_autoencoder()
