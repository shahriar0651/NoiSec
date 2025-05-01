import pyod
from pyod.models.ecod import ECOD
from pyod.models.pca import PCA
from pyod.models.kpca import KPCA
from pyod.models.ocsvm import OCSVM
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.deep_svdd import DeepSVDD
from pyod.models.copod import COPOD
from pyod.models.gmm import GMM
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.hbos import HBOS
from sklearn.semi_supervised import LabelSpreading



def get_baseline_model(cfg, model_name, n_components):

    # num_jobs  = cfg.models.num_jobs
    # num_jobs  = cfg.models.num_jobs
    # random_state  = cfg.models.random_state
    # contamination  = cfg.models.contamination
    # epochs  = cfg.models.epochs
    # n_neighbors = cfg.models.n_neighbors

    # if cfg.rep == 'Feat':
    #     hidden_neurons = cfg.models.hidden_neurons_feat
    # elif cfg.rep == 'Conf':
    #     hidden_neurons = cfg.models.hidden_neurons_conf
    
    contamination  = cfg.models.contamination
    if model_name == 'Manda':
        return LabelSpreading(gamma=6)

    if model_name == 'GMM':
        return GMM(n_components= n_components, #5, #1, #n_components,
                   covariance_type="full",
                   tol=1e-3,
                   reg_covar=1e-6,
                   max_iter=100,
                   n_init=1,
                   init_params="kmeans",
                   weights_init=None,
                   means_init=None,
                   precisions_init=None,
                   random_state=None,
                   warm_start=False,
                   contamination=contamination)
    
   
    print(f"\n\n\nThere is no model for {model_name}")
    return None
    
def get_all_anom_detect_models(cfg, n_components):
    baseline_model_dict = {}
    for model_name in cfg.models.model_List:
        baseline_model_dict[model_name] = get_baseline_model(cfg, model_name, n_components)
    print("Model loaded: ", baseline_model_dict.keys())
    return baseline_model_dict

