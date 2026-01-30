import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from functools import partial
from scipy.ndimage import uniform_filter
from scipy.stats import sem
# from sklearn.preprocessing import normalize
from numpy.linalg import norm
import seaborn as sns
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
sns.set_theme()
sns.set_style("ticks")

# gets all the files from input and checks if they are csv
def get_files(log_folder ='/home/rishika/sim/NAS_rishika/human_pilot_data'):
    files = []
    if type(log_folder)==list:
        for fol in log_folder:
            os.chdir(os.path.realpath(fol))
            for f in os.listdir(os.path.realpath(fol)):
                if f.endswith(".csv"):
                    files.append(f)
    else:
        for f in os.listdir(os.path.realpath(log_folder)):
            os.chdir(os.path.realpath(log_folder))
            if f.endswith(".csv"):
                files.append(f)
    return files

# counts number of participants, provided you've given them correct subject numbers
'''would prefer if this can be linked to the _human_expt_data.xlsx file, and can get max participant number from there
should.nt be too hard'''
def num_participants(files):
    participants = int(files[-1][0])
    return participants
    
# exclude files if required, by filename
'''in future, would prefer if files are excluded by performance rather than by filename, 
need to include script to update this'''
def exclude_files(files, exclusion=None):
    if exclusion != None:
        if type(exclusion) == list:
            for x in exclusion:
                if x in files:
                    files.pop(files.index(x))
                else:
                    pass
        else:
            if exclusion in files:
                    files.pop(files.index(exclusion))
            else:
                pass
    return files
            
# merging all csv for a participant, columns available = used_cols
def merge_files_to_df(files, used_cols = ['key_resp.rt', 'key_resp.keys', 'sess_mean', 
                                          'rew_val','participant', 'session', 'trials.thisN', 'status', 'shuffled']):
    df = pd.concat(map(partial(pd.read_csv, usecols = used_cols), files), ignore_index = True)
    return df

def correct_df(df):
# correcting all rt and key vals
# EXTREMELY SLOW - is there no better way of doing this? ### ASK OTHER PEOPLE ### RELATIVELY OKAY NOW
    pd.options.mode.chained_assignment = None  # default='warn'
    for i in range(df.shape[0]): 
        try:
            df['key_resp.rt'][i] = float(df['key_resp.rt'].values[i].strip('[""]'))
            df['key_resp.keys'][i] = df['key_resp.keys'].values[i].strip('[""]').replace('\'', '')
        except:
            pass
    pd.options.mode.chained_assignment = 'warn'  # default='warn'
    return df 


def construct_dict(exclusions, conditions = None):
    full_dict = {}
    files = get_files('L:/human_pilot_data')
    files.sort()
    participants = num_participants(files)
    for participant in range(1, participants+1):
        if conditions=='str':
            temp = ([participant_file for participant_file in files if (participant_file.startswith(str(participant)) & ("unstr" not in participant_file))])
        elif conditions == 'unstr':
            temp = ([participant_file for participant_file in files if (participant_file.startswith(str(participant)) & ("unstr" in participant_file))])
        else:
            temp = [participant_file for participant_file in files if (participant_file.startswith(str(participant)))]
        temp_df = correct_df(merge_files_to_df(exclude_files(temp, exclusions)))
        full_dict[participant] = temp_df
    return full_dict
    