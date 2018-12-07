#!/usr/bin/env python3
"""Script for performing a grid search."""

import sys
from testing_framework import Testing_Framework
from create_data import Distribution
from models import Learned_FC, Learned_Res, Learned_Bits, BTree, Hybrid, Hybrid_Original
from multiprocessing import Pool


def run_test(model, mp, tp):
    m = model(**mp)
    tf = Testing_Framework(model=m, **tp)
    tf.run_tests()
    tf.save_results()


def main(argv):

    # Constant Parameters
    testing_params = {
        'sample_size': 100000,
        'inference_samples': 10,
        'train_percent': 1.0,
        'num_tests': 3,
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
        ('learned_bits', Learned_Bits),
        ('btree', BTree),
        # ('hybrid', Hybrid),
        # ('hybrid_orig', Hybrid_Original),
    ]

    model_constant_params = {
        'learned_fc': {'epochs': 100, 'batch_size': 10000},
        'learned_res': {'epochs': 100, 'batch_size': 10000},
        'learned_bits': {'epochs': 100, 'batch_size': 10000},
        'btree': {},
        # 'hybrid': {},
        # 'hybrid_orig': {},
    }

    model_grid_params = {
        'learned_fc': [],
        'learned_res': [],
        'learned_bits': [],
        'btree': [{}],
        # 'hybrid': [{}],
        # 'hybrid_orig': [{}],
    }

    # Create tests for learned model
    for loss in ['mean_squared_error', 'mean_absolute_error']:
        for optimizer in ['adam', 'adadelta']:
            for activation in ['relu', 'linear']:
                for hidden_layers in [[10,]*2, [10,]*8, [100,]*2, [100,]*8, [1000,]*2]:
                    structure = []
                    for num_neurons in hidden_layers:
                        structure.append({
                            'activation': activation,
                            'hidden': num_neurons,
                        })

                    model_grid_params['learned_fc'].append({
                        'network_structure': structure, 
                        'loss': loss, 
                        'optimizer': optimizer, 
                        })
                    model_grid_params['learned_res'].append({
                        'network_structure': structure, 
                        'loss': loss, 
                        'optimizer': optimizer, 
                        })
                    model_grid_params['learned_bits'].append({
                        'network_structure': structure, 
                        'loss': loss, 
                        'optimizer': optimizer, 
                        })

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
        print("Starting Test {} of {}: {}, {}, {}".format(i, len(grid_search), model, sorted(mp.items()), sorted(tp.items())))
        try:
            run_test(model, mp, tp)
        except Exception as e:
            print("****Test failed****:", repr(e))

    # # Perform the grid search using a pool
    # p = Pool(4)
    # p.starmap(run_test, grid_search)


if __name__ == '__main__':
    main(sys.argv)
