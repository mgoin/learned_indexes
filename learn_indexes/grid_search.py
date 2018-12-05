#!/usr/bin/env python3
"""Script for performing a grid search."""

import sys
from testing_framework import Testing_Framework
from create_data import Distribution
from models import Learned_Model, BTree, Hybrid, Hybrid_Original


def run_test(model, mp, tp, number_of_tests):
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
        ('learned_model', Learned_Model),
        ('btree', BTree),
        ('hybrid', Hybrid),
        ('hybrid_orig', Hybrid_Original),
    ]

    model_constant_params = {
        'learned_model': {
            'hidden_activation': 'relu',
            'training_method': 'full',
            'search_method': 'linear',
            'batch_size': 100000,
            'epochs': 4,
        },
        'btree': {},
        'hybrid': {},
        'hybrid_orig': {},
    }

    model_grid_params = {
        'learned_model': [],
        'btree': [{}],
        'hybrid': [{}],
        'hybrid_orig': [{}],
    }

    # Create tests for learned model
    for network_type in ['fc', 'res']:
        for hidden_layers in [[20,]*2, [500,]*4,]:
            parms = {
                'network_type': network_type,
                'hidden_layers': hidden_layers,
            }
            model_grid_params['learned_model'].append(parms)

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

    # Perform grid search
    for i, (model, mp, tp) in enumerate(grid_search):
        print("Test {} of {}: {}, {}, {}".format(i, len(grid_search), model, sorted(mp.items()), sorted(tp.items())))
        run_test(model, mp, tp, number_of_tests)


if __name__ == '__main__':
    main(sys.argv)
