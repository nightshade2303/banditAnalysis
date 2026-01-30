from opconNosepokeFunctions import *
from supplementaryFunctions import *
from scipy.optimize import minimize
from scipy.stats import entropy
from scipy.stats import ttest_rel
from sklearn.linear_model import LogisticRegression
from scipy.ndimage import gaussian_filter1d

def data_prep(dataset, hist = 20, trialsinsess = 100, task = 'unstr', head = False):
    dataset = dataset[dataset.task == task].groupby(['animal','session']).filter(lambda x: x.reward.size >= trialsinsess)
    dataset['ct0'] = dataset.port.values
    for i in range(1,hist): 
        dataset['ct'+str(i)] = dataset.groupby(['animal','session']).port.shift(i)
        dataset['shift_t'+str(i-1)] = dataset['ct'+str(i)]==dataset['ct'+str(i-1)]
        dataset['shift_t'+str(i-1)] = dataset['shift_t'+str(i-1)].replace({True: 0, False: 1})
        dataset['rt'+str(i)] = dataset.groupby(['animal','session']).reward.shift(i)
        dataset['rt'+str(i)] = dataset['rt'+str(i)]#.replace({0:-1})
#         dataset['choice_t'+str(i)] = dataset['choice_t'+str(i)].replace({1:'a', 2:'b', 3:'c', 4:'d'})
    dataset = dataset.dropna()
    if head == True:
        dataset = dataset.groupby(['animal','session']).head(trialsinsess)

    return dataset

