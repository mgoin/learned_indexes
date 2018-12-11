import utils.datastore as ds
import numpy as np
import sys
import pandas as pd


def avg_results(data, result):
    """Average the results for each run to get a single data point."""

    result['train_time'] = np.mean(result['train_time'])
    result['insert_time'] = np.mean(result['insert_time'])

    predict, get = zip(*result['pre_insert_inference_time'])
    del result['pre_insert_inference_time']
    result['pre_insert_inference_time_predict'] = np.mean([predict])
    result['pre_insert_inference_time_get'] = np.mean([get])

    try:
        predict, get = zip(*result['post_insert_inference_time'])
    except:
        predict = []
        get = []
    del result['post_insert_inference_time']
    result['post_insert_inference_time_predict'] = np.mean([predict])
    result['post_insert_inference_time_get'] = np.mean([get])

    result['pre_insert_mean_error'] = np.mean(result['pre_insert_mean_error'])
    result['pre_insert_min_error'] = np.mean(result['pre_insert_min_error'])
    result['pre_insert_max_error'] = np.mean(result['pre_insert_max_error'])
    result['post_insert_mean_error'] = np.mean(result['post_insert_mean_error'])
    result['post_insert_min_error'] = np.mean(result['post_insert_min_error'])
    result['post_insert_max_error'] = np.mean(result['post_insert_max_error'])

    data.append(result)

    return data


def expand_results(data, result):
    """Expand results by creating a data entry for run."""

    num_tests = len(result['train_time'])

    for i in range(num_tests):
        r = result.copy()
        r['train_time'] = r['train_time'][i]

        try:
            r['insert_time'] = r['insert_time'][i]
        except:
            pass

        predict, get = zip(*r['pre_insert_inference_time'])
        del r['pre_insert_inference_time']
        r['pre_insert_inference_time_predict'] = predict[i]
        r['pre_insert_inference_time_get'] = get[i]

        try:
            try:
                predict, get = zip(*r['post_insert_inference_time'])
            except:
                predict = []
                get = []
            del r['post_insert_inference_time']
            r['post_insert_inference_time_predict'] = predict[i]
            r['post_insert_inference_time_get'] = get[i]
        except:
            pass

        r['pre_insert_mean_error'] = r['pre_insert_mean_error'][i]
        r['pre_insert_min_error'] = r['pre_insert_min_error'][i]
        r['pre_insert_max_error'] = r['pre_insert_max_error'][i]

        try:
            r['post_insert_mean_error'] = r['post_insert_mean_error'][i]
            r['post_insert_min_error'] = r['post_insert_min_error'][i]
            r['post_insert_max_error'] = r['post_insert_max_error'][i]
        except:
            pass

        r['training_history'] = r['training_history'][i]

        data.append(r)

    return data


def get_panda_from_results(path='../results', run_handling='expand'):
    """Read the results data and store it in a panda dataframe

    Parameters:
        path (string) - Path to the results directory.
        run_handling (string) - method used to handle entires with multiple runs stored.

    Returns:
        A panda dataframe.

    """

    results = ds.read_all_data_from_folder(path)
    data_tmp = []
    for result in results:
        if run_handling == 'expand':
            data_tmp = expand_results(data_tmp, result)
        elif run_handling == 'avg':
            data_tmp = avg_results(data_tmp, result)
        else:
            raise Exception('Unknown run_handling: {}'.format(run_handling))

    data = pd.DataFrame(data_tmp)
    del data_tmp
    return data


def save_as_csv(path):
    """Save the results to a csv file."""

    data = get_panda_from_results()
    data.to_csv(path)



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
