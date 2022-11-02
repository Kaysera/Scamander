import sys
sys.path.append("F:\\guillermo\\Documentos\\Universidad\\Doctorado\\Articulos\\Fuzzy LORE\\Scamander")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import logging
import random
from joblib import Parallel, delayed
import pickle
import datetime
import numpy as np
import pandas as pd

from keras.models import load_model
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import pdist, squareform

from sace.blackbox import BlackBox

from alibi.explainers import CEM

from cf_eval.metrics import *

from experiments.config import *
from experiments.util import get_tabular_dataset
import hydra
from omegaconf import DictConfig, OmegaConf


def experiment(cfe, bb, X_train, variable_features, metric,
               X_test, nbr_test, search_diversity, dataset, black_box, known_train,
               variable_features_flag, experiment_id):
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()

    conf = {
        'kappa': np.random.randint(1, 20) * 0.01,
        'beta': np.random.randint(1, 40) * 0.01,
        'c_init': np.random.randint(1, 10),
        'c_steps': np.random.randint(10, 50),
        'lr_init': 1e-2
    }

    total_cf = 0
    mode = 'PN'  # 'PN' (pertinent negative) or 'PP' (pertinent positive)
    shape = (1,) + X_train.shape[1:]  # instance shape
    kappa = conf['kappa'] # minimum difference needed between the prediction probability for the perturbed instance on the
    # class predicted by the original instance and the max probability on the other classes
    # in order for the first loss term to be minimized
    beta = conf['beta']  # weight of the L1 loss term
    c_init = conf['c_init']  # initial weight c of the loss term encouraging to predict a different class (PN) or
    # the same class (PP) for the perturbed instance compared to the original instance to be explained
    c_steps = conf['c_steps']  # nb of updates for c
    max_iterations = 500  # nb of iterations per value of c
    clip = (-1000., 1000.)  # gradient clipping
    lr_init = conf['lr_init']  # initial learning rate

    predict_fn = lambda x: bb.predict_proba(x)  # only pass the predict fn which takes numpy arrays to CEM

    index_test_instances = np.random.choice(range(len(X_test)), nbr_test)

    for test_id, i in enumerate(index_test_instances):
        logging_message = f'Dataset: {dataset}, blackbox: {black_box}, exp_id: {experiment_id}, progress: {(test_id + 1) / len(index_test_instances)}'
        logging.info(logging_message)
        try:
            x = X_test[i]

            if variable_features_flag:
                feature_range = (X_train.min(axis=0).reshape(shape),  # feature range for the perturbed instance
                                X_train.max(axis=0).reshape(shape))  # can be either a float or array of shape (1xfeatures)

                feature_range[0][:, variable_features] = x[variable_features]
                feature_range[1][:, variable_features] = x[variable_features]

                # initialize CEM explainer and explain instance
                exp = CEM(predict_fn, mode, shape, kappa=kappa, beta=beta, feature_range=feature_range,
                        max_iterations=max_iterations, c_init=c_init, c_steps=c_steps,
                        learning_rate_init=lr_init, clip=clip)
                exp.fit(X_train, no_info_type='median')  # we need to define what feature values contain the least
                # info wrt predictions
                # here we will naively assume that the feature-wise median
                # contains no info; domain knowledge helps!

            explanation = exp.explain(x.reshape(1, -1), verbose=False)
            cf_list = explanation.PN
            if cf_list is None:
                logging.info('No counterfactual found')

                cf_list = np.array([])
            else:
                logging.info('Counterfactual found')

                total_cf += 1
        except Exception as e:
            logging_message = f'Dataset: {dataset}, blackbox: {black_box}, exp_id: {experiment_id}, instance: {test_id}'
            logging.error(logging_message)
            logging.error(e.with_traceback())
    
    with open(path_results + 'paramsearch_%s_%s_%s_%s.csv' % (dataset, black_box ,experiment_id, cfe), 'a+') as f:
        f.write(f"{total_cf/nbr_test},{conf['kappa']},{conf['beta']},{conf['c_init']},{conf['c_steps']},{conf['lr_init']}\n")

    return total_cf


def main(dataset, black_box, random_state):

    logging_message = f'Launching experiment: {black_box}-{dataset}-{random_state}'
    logging.info(logging_message)
    nbr_test = 15
    normalize = 'standard'

    known_train = True
    search_diversity = False
    metric = 'none'
    variable_features_flag = True
    random.seed(random_state)
    np.random.seed(random_state)


    if dataset not in dataset_list:
        print('unknown dataset %s' % dataset)
        return -1

    if black_box not in blackbox_list:
        print('unknown black box %s' % black_box)
        return -1

    # if cfe not in cfe_list:
    #     print('unknown counterfactual explainer %s' % cfe)
    #     return -1

    data = get_tabular_dataset(dataset, path_dataset, normalize=normalize, test_size=test_size,
                            random_state=random_state, encode=None if black_box == 'LGBM' else 'onehot')
    X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']

    variable_features = data['variable_features']



    if black_box in ['DT', 'RF', 'SVM', 'NN', 'LGBM']:
        bb = pickle.load(open(path_models + '%s_%s.pickle' % (dataset, black_box), 'rb'))
        # if black_box == 'RF':
        #     bb.n_jobs = 5
    elif black_box in ['DNN']:
        bb = load_model(path_models + '%s_%s.h5' % (dataset, black_box))
    else:
        print('unknown black box %s' % black_box)
        raise Exception

    bb = BlackBox(bb)


    experiment('cem', bb, X_train, variable_features, metric,
        X_test, nbr_test, search_diversity, dataset, black_box, known_train,
        variable_features_flag, random_state)




@hydra.main(version_base=None, config_path="../conf", config_name="config")
def my_app(cfg : DictConfig) -> None:
    main(cfg.dataset.name, cfg.bb.name, cfg.rs.value)


if __name__ == "__main__":
    my_app()


