import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

from pathlib import Path
import time
from joblib import dump, load
import json
from helper import *
from sklearn.base import clone

def train_all_anom_detect_models(cfg, baseline_model_dict, X_train, Y_train, attack):
    if Y_train.ndim==2:
       Y_train = np.argmax(Y_train, axis=1)
    print("Y_train shape: ", Y_train.shape)
    model_root_dir = cfg.models_dir
    model_type = cfg.models.model_type
    dataset = cfg.dataset.name
    num_clasess = cfg.dataset.num_classes

    training_time_dict = {}
    
    for model_name, model_init in baseline_model_dict.items():
        print("model_name : ", model_name)
        if model_name in cfg.models.model_List:
            model_init.fit(X_train)
            baseline_model_dict[model_name] = model_init
            model_dir = Path(f'{model_root_dir}/{model_type}/{cfg.scale}/{cfg.rep}/{model_type}_{dataset}_{num_clasess}_{model_name}_{attack}')
            model_dir.parent.mkdir(exist_ok=True, parents=True)
            dump(model_init, model_dir)
            if cfg.verbose:
                print("Model name: ", model_name)
                print(f"{model_name} trained and saved!")

    print(f"Training complete!")
    return baseline_model_dict

def load_all_anom_detect_models(cfg, attack):
    
    model_root_dir = cfg.models_dir
    model_type = cfg.models.model_type
    num_clasess = cfg.dataset.num_classes
    dataset = cfg.dataset.name

    baseline_model_dict = {}
    for model_name in cfg.models.model_List:
        if model_name in cfg.models.model_List:
            model_dir = Path(f'{model_root_dir}/{model_type}/{cfg.scale}/{cfg.rep}/{model_type}_{dataset}_{num_clasess}_{model_name}_{attack}')
            baseline_model_dict[model_name] = load(model_dir)
        if cfg.verbose:
            print(f"{model_name} loaded !")
    return baseline_model_dict