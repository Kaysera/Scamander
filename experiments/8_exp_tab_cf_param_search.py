import sys
sys.path.append("F:\\guillermo\\Documentos\\Universidad\\Doctorado\\Articulos\\Fuzzy LORE\\Scamander")

import pickle
import datetime
import numpy as np
from joblib import Parallel, delayed

from keras.models import load_model


from sace.blackbox import BlackBox

from alibi.explainers import Counterfactual

from cf_eval.metrics import *

from experiments.config import *
from experiments.util import get_tabular_dataset

# Watcher method


def experiment(cfe, bb, X_train, variable_features, metric,
               X_test, nbr_test, search_diversity, dataset, black_box, known_train, exp_id):

    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()
    shape = (1,) + X_train.shape[1:]

    conf = {
        'tol': np.random.randint(1, 10) * 0.01,
        'lam_init': np.random.randint(1, 10) * 0.1,
        'max_lam_steps': np.random.randint(10, 50),
        'learning_rate_init': np.random.randint(1, 20) * 0.01
    }

    target_proba = 0.5
    tol = conf['tol']  # want Counterfactuals with p(class)>0.99
    target_class = 'other'
    max_iter = 1000
    lam_init = conf['lam_init']
    max_lam_steps = conf['max_lam_steps']
    learning_rate_init = conf['learning_rate_init']

    total_cf = 0

    predict_fn = lambda x: bb.predict_proba(x)

    index_test_instances = np.random.choice(range(len(X_test)), nbr_test)

    print(datetime.datetime.now(), dataset, black_box, cfe, metric, known_train, search_diversity)

    for test_id, i in enumerate(index_test_instances):
        try:
            print(datetime.datetime.now(), dataset, black_box, cfe, test_id, len(index_test_instances),
                '%.2f' % (test_id / len(index_test_instances)))
            x = X_test[i]

            feature_range = (X_train.min(axis=0).reshape(shape),  # feature range for the perturbed instance
                                X_train.max(axis=0).reshape(shape))  # can be either a float or array of shape (1xfeatures)

            feature_range[0][:, variable_features] = x[variable_features]
            feature_range[1][:, variable_features] = x[variable_features]

            # initialize explainer
            exp = Counterfactual(predict_fn, shape=shape, target_proba=target_proba, tol=tol,
                                target_class=target_class, max_iter=max_iter, lam_init=lam_init,
                                max_lam_steps=max_lam_steps, learning_rate_init=learning_rate_init,
                                feature_range=feature_range)

            explanation = exp.explain(x.reshape(1, -1))
            if explanation.cf is not None:
                cf_list = explanation.cf['X']
                total_cf += 1
            else:
                cf_list = np.array([])
        except:
            with open(path_results + 'error_%s_%s_%s_cfw.csv' % (dataset, black_box ,exp_id), 'a+') as f:
                f.write(f"{index_test_instances},{conf['tol']},{conf['lam_init']},{conf['max_lam_steps']},{conf['learning_rate_init']}\n")
            print('No counterfactual found')


    with open(path_results + 'paramsearch_%s_%s_%s_cfw.csv' % (dataset, black_box ,exp_id), 'a+') as f:
        f.write(f"{total_cf/nbr_test},{conf['tol']},{conf['lam_init']},{conf['max_lam_steps']},{conf['learning_rate_init']}\n")

    return total_cf

def main():

    nbr_test = 10
    datasets = ['compas', 'adult', 'fico', 'german']
    black_boxes = ['SVM', 'NN']
    normalize = 'standard'


    known_train = True
    search_diversity = False
    metric = 'none'

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
            #     print('unknown Counterfactual explainer %s' % cfe)
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

            Parallel(n_jobs=10)(
                delayed(experiment)('cfw', bb, X_train, variable_features, metric,
                    X_test, nbr_test, search_diversity, dataset, black_box, known_train, i)
                for i in range(100)
            )
            
            


if __name__ == "__main__":
    main()


