#!/usr/bin/env python3
"""Script for performing a grid search."""

import sys
from testing_framework import Testing_Framework
from create_data import Distribution
from models import Learned_FC, Learned_Res, BTree, Hybrid, Hybrid_Original
from multiprocessing import Pool


def run_test(model, mp, tp, number_of_tests=1):
    m = model(**mp)
    tf = Testing_Framework(model=m, **tp)
    tf.run_tests(number_of_tests)
    tf.save_results()


def main(argv):

    # Constant Parameters
    number_of_tests = 1
    testing_params = {
        'sample_size': 100,
        'inference_samples': 100,
        'train_percent': 0.8,
    }

    distributions = [
        Distribution.RANDOM,
        Distribution.EXPONENTIAL,
        Distribution.NORMAL,
        Distribution.LOGNORMAL,
    ]

    models = [
        ('learned_fc', Learned_FC),
        ('learned_res', Learned_Res),
        ('btree', BTree),
        ('hybrid', Hybrid),
        ('hybrid_orig', Hybrid_Original),
    ]

    model_constant_params = {
        'learned_fc': {},
        'learned_res': {},
        'btree': {},
        'hybrid': {},
        'hybrid_orig': {},
    }

    model_grid_params = {
        'learned_fc': [],
        'learned_res': [],
        'btree': [{}],
        'hybrid': [{}],
        'hybrid_orig': [{}],
    }

    # Create tests for learned model
    for hidden_layers in [[20,]*2, [500,]*4,]:
        structure = []
        for num_neurons in hidden_layers:
            structure.append({
                'activation': 'relu',
                'hidden': num_neurons,
            })
        model_grid_params['learned_fc'].append({'network_structure': structure,})

    for hidden_layers in [[20,]*2, [500,]*4,]:
        structure = []
        for num_neurons in hidden_layers:
            structure.append({
                'activation': 'relu',
                'hidden': num_neurons,
            })
        model_grid_params['learned_res'].append({'network_structure': structure,})

    grid_search = []
    # Construct grid search
    for distribution in distributions:
        for model_key, model in models:
            for model_params in model_grid_params[model_key]:
                mp = model_constant_params[model_key].copy()
                mp.update(model_params)

                tp = testing_params.copy();
                tp.update({'distribution': distribution})

                grid_search.append((model, mp, tp))

    # Perform grid search sequentially
    for i, (model, mp, tp) in enumerate(grid_search):
        print("Test {} of {}: {}, {}, {}".format(i, len(grid_search), model, sorted(mp.items()), sorted(tp.items())))
        run_test(model, mp, tp, number_of_tests)

    # # Perform the grid search using a pool
    # p = Pool(4)
    # p.starmap(run_test, grid_search)


if __name__ == '__main__':
    main(sys.argv)
