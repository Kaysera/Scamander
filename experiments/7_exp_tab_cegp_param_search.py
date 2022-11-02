import sys
sys.path.append("F:\\guillermo\\Documentos\\Universidad\\Doctorado\\Articulos\\Fuzzy LORE\\Scamander")


import os
from joblib import Parallel, delayed

import pickle
import datetime
import numpy as np
import pandas as pd

from keras.models import load_model

from sace.blackbox import BlackBox

from alibi.explainers import CounterfactualProto

from cf_eval.metrics import *

from experiments.config import *
from experiments.util import get_tabular_dataset


def experiment(cfe, bb, X_train, variable_features, metric,
               X_test, nbr_test, search_diversity, dataset, black_box, known_train,
               variable_features_flag):
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()
    conf = {
        'beta': np.random.randint(1, 40) * 0.01,
        'c_init': np.random.randint(1, 10),
        'c_steps': np.random.randint(10, 50),
    }


    shape = (1,) + X_train.shape[1:]  # instance shape
    max_iterations = 500
    total_cf = 0
    beta = conf['beta']  # weight of the L1 loss term
    c_init = conf['c_init']  # initial weight c of the loss term encouraging to predict a different class (PN) or
    c_steps = conf['c_steps']  # nb of updates for c
    max_iterations = 500  # nb of iterations per value of c

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

                exp = CounterfactualProto(predict_fn,
                                        shape,
                                        beta=beta,
                                        # cat_vars=cat_vars_ohe,
                                        # ohe=True,  # OHE flag
                                        max_iterations=max_iterations,
                                        feature_range=feature_range,
                                        c_init=c_init,
                                        c_steps=c_steps
                                        )

                exp.fit(X_train, d_type='abdm', disc_perc=[25, 50, 75])

            explanation = exp.explain(x.reshape(1, -1))

            if explanation.cf is not None:
                cf_list = explanation.cf['X']
                total_cf += 1
            else:
                cf_list = np.array([])
        except:
            print('No counterfactual found')


    with open(path_results + 'paramsearch_%s_%s_cegp.csv' % (dataset, black_box), 'a+') as f:
        f.write(f"{total_cf/nbr_test},{conf['beta']},{conf['c_init']},{conf['c_steps']}\n")

    return total_cf

def main():

    nbr_test = 10
    datasets = ['german', 'compas', 'adult', 'fico']
    black_boxes = ['SVM', 'NN']
    normalize = 'standard'

    # dataset = sys.argv[1]
    # black_box = sys.argv[2]
    # cfe = sys.argv[3]
    # nbr_test = 100 if len(sys.argv) < 5 else int(sys.argv[4])
    known_train = True
    search_diversity = False
    metric = 'none'
    variable_features_flag = True

    np.random.seed(random_state)

    for dataset in datasets:
        for black_box in black_boxes:

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
            # experiment('cegp', bb, X_train, variable_features, metric,
            #         X_test, nbr_test, search_diversity, dataset, black_box, known_train,
            #         variable_features_flag)
            Parallel(n_jobs=10)(
                delayed(experiment)('cegp', bb, X_train, variable_features, metric,
                    X_test, nbr_test, search_diversity, dataset, black_box, known_train,
                    variable_features_flag)
                for i in range(100)
            )


if __name__ == "__main__":
    main()


