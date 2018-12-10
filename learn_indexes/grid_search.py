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
        'train_percent': 0.8,
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
        ('hybrid', Hybrid),
        ('hybrid_orig', Hybrid_Original),
    ]

    mcp = {'epochs': 100, 'batch_size': 10000,
           'loss': 'mean_squared_error', 'optimizer': 'adam'}
    model_constant_params = {
        'learned_fc': mcp,
        'learned_res': mcp,
        'learned_bits': mcp,
        'btree': {},
        'hybrid': mcp,
        'hybrid_orig': {},
    }

    model_grid_params = {
        'learned_fc': [],
        'learned_res': [],
        'learned_bits': [],
        'btree': [{}],
        'hybrid': [],
        'hybrid_orig': [{}],
    }

    # Create tests for learned model
    for training_method in ['start_from_scratch', 'start_from_previous', 'train_only_new']:
        for activation in ['relu', 'linear']:
            for hidden_layers in [[100,]*4, [100,]*8, [100,]*12, [100,]*16]:
                structure = []
                for num_neurons in hidden_layers:
                    structure.append({
                        'activation': activation,
                        'hidden': num_neurons,
                    })

                model_grid_params['learned_fc'].append({'network_structure': structure, 'training_method': training_method})
                model_grid_params['learned_res'].append({'network_structure': structure, 'training_method': training_method})
                model_grid_params['learned_bits'].append({'network_structure': structure, 'training_method': training_method})

    for activation in ['relu', 'linear']:
        for hidden_layers in [[100,]*4, [100,]*8, [100,]*12, [100,]*16]:
            structure = []
            for num_neurons in hidden_layers:
                structure.append({
                    'activation': activation,
                    'hidden': num_neurons,
                })

            for stage in ([1, 2], [1, 5], [1, 10], [1, 5, 10]):
                model_grid_params['hybrid'].append({'network_structure': structure, 'training_method': 'start_from_scratch', 'stage_nums': stage})

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
