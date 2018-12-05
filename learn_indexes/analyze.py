import utils.datastore as ds
import numpy as np
import sys
import pandas as pd


def avg_results(result):
    result['train_time'] = np.mean(result['train_time'])
    result['insert_time'] = np.mean(result['train_time'])

    predict, get = zip(*result['pre_insert_inference_time'])
    del result['pre_insert_inference_time']
    result['pre_insert_inference_time_predict'] = np.mean([predict])
    result['pre_insert_inference_time_get'] = np.mean([get])

    predict, get = zip(*result['post_insert_inference_time'])
    del result['post_insert_inference_time']
    result['post_insert_inference_time_predict'] = np.mean([predict])
    result['post_insert_inference_time_get'] = np.mean([get])

    result['pre_insert_mean_error'] = np.mean(result['pre_insert_mean_error'])
    result['pre_insert_min_error'] = np.mean(result['pre_insert_min_error'])
    result['pre_insert_max_error'] = np.mean(result['pre_insert_max_error'])
    result['post_insert_mean_error'] = np.mean(result['post_insert_mean_error'])
    result['post_insert_min_error'] = np.mean(result['post_insert_min_error'])
    result['post_insert_max_error'] = np.mean(result['post_insert_max_error'])


def get_panda_from_results():
    results = ds.read_all_data_from_folder('../results')
    data_tmp = []
    for result in results:
        avg_results(result)
        data_tmp.append(result)

    data = pd.DataFrame(data_tmp)
    del data_tmp
    return data


def main(argv):
    data = get_panda_from_results()
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('max_colwidth', -1)
    print(data.head(3))
    print(data.describe())
    print(data.dtypes)


if __name__ == '__main__':
    main(sys.argv)
