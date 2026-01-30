import numpy as np
import csv
def calc_prob(pk):
    # calc prob of actions
    unique, counts = np.unique(np.array(pk), return_counts =True)
    outcomes = len(pk)
    return counts/outcomes

# adds two unequal numpy vectors a and b to return c e.g. if a = [20,40,60], b= [10,20,40, 60], c=[30, 60, 100, 60]
def add_uneq_vec(a,b):
    if len(a) < len(b):
        c = b.copy()
        c[:len(a)] += a
    else:
        c = a.copy()
        c[:len(b)] += b
    return c

# sorts dictionary y alphabetical order of keys
def sort_dict(myDict):
    myKeys = list(myDict.keys())
    myKeys.sort()
    sorted_dict = {i: myDict[i] for i in myKeys}
    return sorted_dict

# function for matlab-like tic and toc
import time
def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )
    return "Elapsed time: %f seconds.\n" %tempTimeInterval

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)

# saves a dict of dataframes into a csv file for easy access
def saver(dictex, fname):
    for ind, (key, val) in enumerate(dictex.items()):
        val.to_csv(f'/home/rishika/sim/{ind}_{fname}.csv')

    with open("/home/rishika/sim/keys.txt", "w") as f: #saving keys to file
        f.write(str(list(dictex.keys())))

# loads the dict of dataframess from csv
def loader(name):
    """Reading data from keys"""
    with open("/home/rishika/sim/keys.txt", "r") as f:
        keys = eval(f.read())

    dictex = {}    
    for ind, key in enumerate(keys):
        dictex[key] = pd.read_csv(f"/home/rishika/sim/{ind}_{name}.csv".format(str(key)))
        dictex[key] = dictex[key].drop(['Unnamed: 0'], axis = 1)
    
    return dictex