import numpy as np
import pandas as pd
def session_averager(df, mask = None, metric = None, tasks = None):
    if metric is None:
        assert isinstance(metric, str), "pass a metric column"

    if tasks is not None:
        assert isinstance(tasks, list), "tasks should be list"

        metric_dict = {}
        for task in tasks:
<<<<<<< HEAD
            for ind, (an, group) in enumerate(df.groupby('animal')):
=======
            for ind, (an, group) in enumerate(df[df.task == task].groupby('animal')):
>>>>>>> 80714cd (initial commit)
                L = subset_metric_to_numpy(group, metric)
                metric_dict[an, task] = np.mean(L, axis = 0)

    else:
        metric_dict = {}
        for ind, (an, group) in enumerate(df.groupby('animal')):
            L = subset_metric_to_numpy(group, metric)
            metric_dict[an] = np.mean(L, axis = 0)
            
    return metric_dict


def subset_metric_to_numpy(group, metric):
    # works only for equal trials in all sessions
    g = group.groupby('session').cumcount()
    L = (np.array(group.set_index(['session', g])
            .unstack(fill_value=0)
            .stack(future_stack = True).groupby(level=0)
            .apply(lambda x: x[metric].values.tolist())
            .tolist()))
    return L

def generate_tm(df, mask = None, col1 = 'port', col2 = 'choice_t1'):
    if mask is not None:
        mat = pd.crosstab(
            df.loc[mask, col1], 
            df.loc[mask, col2], 
            normalize='index'
            )
        return mat
    mat = pd.crosstab(
            df.loc[col1], 
            df.loc[col2], 
            normalize='index'
            )
    return mat

