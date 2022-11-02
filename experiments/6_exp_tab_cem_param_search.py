import sys
sys.path.append("F:\\guillermo\\Documentos\\Universidad\\Doctorado\\Articulos\\Fuzzy LORE\\Scamander")
import os
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

    print(datetime.datetime.now(), dataset, black_box, cfe, metric, known_train, search_diversity)

    for test_id, i in enumerate(index_test_instances):
        try:
            print(datetime.datetime.now(), dataset, black_box, cfe, test_id, len(index_test_instances),
                '%.2f' % (test_id / len(index_test_instances)))
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
                cf_list = np.array([])
            else:
                total_cf += 1
        except:
            print('No PN found')
    
    with open(path_results + 'paramsearch_%s_%s_%s_cem.csv' % (dataset, black_box, experiment_id), 'a+') as f:
        f.write(f"{total_cf/nbr_test},{conf['kappa']},{conf['beta']},{conf['c_init']},{conf['c_steps']},{conf['lr_init']}\n")

    return total_cf


def main():

    nbr_test = 10
    datasets = ['adult', 'fico']
    black_boxes = ['SVM', 'NN']
    normalize = 'standard'

    known_train = True
    search_diversity = False
    metric = 'none'
    variable_features_flag = True

    np.random.seed(random_state)

    for dataset in datasets:
        for black_box in black_boxes:
            if dataset not in dataset_list:
                print('unknown dataset %s' % dataset)
                return -1

            if black_box not in blackbox_list:
                print('unknown black box %s' % black_box)
                return -1

            # if cfe not in cfe_list:
            #     print('unknown counterfactual explainer %s' % cfe)
            #     return -1

            print(datetime.datetime.now(), dataset, black_box)

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

            config = [
                {
                    'kappa': kappa,
                    'beta': beta,
                    'c_init': c_init,
                    'c_steps': c_steps,
                    'lr_init': lr_init
                }
                for kappa in [0.01, 0.05, 0.1, 0.15, 0.2]
                for beta in [0.01, 0.1, 0.2, 0.4]
                for c_init in [1, 2, 5, 8, 10]
                for c_steps in [10, 30, 50]
                for lr_init in [1e-2, 1e-3, 1e-1]
            ]

            Parallel(n_jobs=10)(
                delayed(experiment)('cem', bb, X_train, variable_features, metric,
                    X_test, nbr_test, search_diversity, dataset, black_box, known_train,
                    variable_features_flag, i)
                for i in range(100)
            )




if __name__ == "__main__":
    main()


