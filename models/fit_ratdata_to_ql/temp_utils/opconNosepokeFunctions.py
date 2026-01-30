import numpy as np
import pandas as pd
import os
import re
from functools import partial
from temp_utils import plotSettings

# FASTER - 20241126
def get_files(log_folder='L:/Box1/', extension='.dat'):
    """
    Retrieves all files with the specified extension from the provided folder(s).
    """
    if isinstance(log_folder, list):
        return [f for fol in log_folder for f in os.listdir(os.path.realpath(fol)) if f.endswith(extension)]
    return [f for f in os.listdir(os.path.realpath(log_folder)) if f.endswith(extension)]

# def get_files(log_folder ='L:/Box1/', extension = '.dat'):
#     ''' gets all the files from input and checks if they are csv/dat
# '''
#     files = []
#     if type(log_folder)==list:
#         for fol in log_folder:
#             os.chdir(os.path.realpath(fol))
#             for f in os.listdir(os.path.realpath(fol)):
#                 if f.endswith(extension):
#                     files.append(f)
#     else:
#         for f in os.listdir(os.path.realpath(log_folder)):
#             os.chdir(os.path.realpath(log_folder))
#             if f.endswith(extension):
#                 files.append(f)
#     return files
    
# exclude files if required, by expression
def exclude_files(files, exclusion=['Roselia.+.dat', 'test-2022.+.dat', 'test-2023.+.dat', 'Azurill.+.dat']):
    if exclusion != None:
        if type(exclusion) != list:
            excluded = [f for f in files if re.search(exclusion, f)]
        excluded = [f for f in files for x in exclusion if re.search(x, f)]
        
    files = [file for file in files if file not in excluded]
    return files

# MODIFIED 2023-10-25 TO ADD COLUMN NAMES
# merging all csv for a participant, index reset acc to filename
def merge_files_to_df(files):
    names = ['event', 'value', 'sm', 'tp', 'smtime', 'eptime', 'dw', 'ww']
    df = pd.concat(map(partial(pd.read_csv, header=None, names = names), files), ignore_index = True).dropna()
    
    # if 22 exists with the same smtime and eptime as nosepoke (the list of numbers is all nosepokes, Teensy3.6 and 4.1 both)
    # then mark as gpio_lick 0
    # if not then gpio_lick 1
    # rest 5
    
    df['gpio_lick'] = np.ones(df.shape[0])*5
    np_m = (df.event.isin([22, 21, 20, 19, 18, 17, 37])) & (df.duplicated(subset = ['eptime', 'smtime']))
    df.gpio_lick = df['gpio_lick'].mask(np_m, 0)
    lick_m = (df.event.isin([22])) & (df.gpio_lick == 5)
    df.gpio_lick = df['gpio_lick'].mask(lick_m, 1)

    # add camera event info!
    # load the thing called camera df for all animals 
    
    return df

# random function to fullprint (could be used, for e.g for a df that you want to see)
def fullprint(*args, **kwargs):
  from pprint import pprint
  import numpy
  opt = numpy.get_printoptions()
  numpy.set_printoptions(threshold=numpy.inf)
  pprint(*args, **kwargs)
  numpy.set_printoptions(**opt)

def winstay_loseshift(lateSessdf):
    rew_stay, rew_shift, no_rew_stay, no_rew_shift = 0, 0, 0, 0
    reward, no_reward = 0, 0
    prob = [0,0,0,0]
    for trial in lateSessdf['trial#']:
    #     print(trial)
        if (trial+1) < (len(lateSessdf['trial#'])):
            if lateSessdf['reward'][trial]==1:
                reward +=1
                if lateSessdf['port'][trial]==lateSessdf['port'][trial+1]:
                    rew_stay+=1
                    rew_shift+=0
                else:
                    rew_stay+=0
                    rew_shift+=1
            else:
                no_reward+=1
                if lateSessdf['port'][trial]==lateSessdf['port'][trial+1]:
                    no_rew_stay+=1
                    no_rew_shift+=0
                else:
                    no_rew_shift+=1
                    no_rew_stay+=0
    prob[0] = rew_stay/reward
    prob[1] = rew_shift/reward
    prob[2] = no_rew_stay/no_reward
    prob[3] = no_rew_shift/no_reward
    return prob

def set_cwd(path='C:/Users/Rishika/Desktop/lab/ATM_analysis/'):
    os.chdir(os.path.realpath(path))
    
def listdir_fullpath(path):
    return [os.path.join(path, f) for f in os.listdir(path)]

# TESTING
# def trializer_v4(df, list_sessionMarker,
#                  trialStartMarker, trialHitMarker,
#                  trialMissMarker, sessionStartMarker,
#                  trialProbMarker, rewardProbMarker,
#                  lickMarker, animal, arms = 4):
    
#     ''' Function to convert a df of joined .dat files with column names in order :
#     'event', 'value', 'sm', 'tp', 'smtime', 'eptime', 'dw', 'ww'
    
#     INPUTS: df, list of session state machine numbers, trial start marker, trial rewarded marker,
#     trial miss/unrewarded marker, session start marker, trial probability marker, session reward prob marker, 
#     lick marker, animal name, num arms in task
    
#     OUTPUT: single trial-wise dataframe of one animal
#     2024-09-30
#     '''
#     # initialize var
#     sessnum = 0
#     npcounter = 0
#     sess_list = []
#     rewprobl = []
    
#     event_lookup = {
#         trialHitMarker: 1,
#         trialMissMarker: 0,
#     }
#     list_events = [trialStartMarker, trialHitMarker, trialMissMarker, trialProbMarker, sessionStartMarker,
#                    trialProbMarker, rewardProbMarker, lickMarker]
    
#     # filter events as np array to increase speed and avoid repetitive indexing
#     filtered_events = df[(df['sm'].isin(list_sessionMarker)) & (df['event'].isin(list_events))].to_numpy()
#     indices = np.where((filtered_events[:, 0] == trialStartMarker) | (filtered_events[:, 0] == sessionStartMarker) |
#                        (filtered_events[:, 0] == rewardProbMarker))[0]
    
#     for ind in indices:
#         # extract information from the current trialStartMarker
#         trialstart = filtered_events[ind, 4]
#         port = filtered_events[ind, 1]
#         datetime = filtered_events[ind, 5]
#         task = filtered_events[ind, 2]
#         event = filtered_events[ind, 0]
    
#         # select events occurring after trialStartMarker
#         nextEvents = filtered_events[ind:]
    
#         try:
#             # find the index of trialMissMarker or trialHitMarker in nextEvents
#             status_indices = np.where((nextEvents[:, 0] == trialMissMarker) | (nextEvents[:, 0] == trialHitMarker))[0]
#             lick_indices = np.where(((nextEvents[:, 0]) == lickMarker) & (nextEvents[:, 1] == 1))[0]
#             statusIndex = status_indices[0] if status_indices.size > 0 else None
#             lickIndex = lick_indices[lick_indices<status_indices[1]] if status_indices.size > 1 else lick_indices
            
#             ms_licktimes = str(nextEvents[lickIndex][:, 4]).strip('[]')
#             ms_lastlick = nextEvents[lickIndex][:, 4][-1]
#             s_licktimes = str(nextEvents[lickIndex][:, 5]).strip('[]')

#             # find the index of trialProbMarker in nextEvents
#             rewprob_index = np.where(nextEvents[:, 0] == trialProbMarker)[0][0] 
#             #contLine: if np.any(nextEvents[:, 0] == trialProbMarker) else None
    
#             if statusIndex is not None:
#                 # find reward, trialend, and rewprob based on identified events
#                 reward = event_lookup[nextEvents[statusIndex, 0]]
#                 trialend = nextEvents[statusIndex, 4]
#                 rewprob = nextEvents[rewprob_index, 1] if rewprob_index is not None else np.nan
#             else:
#                 # if trialMissMarker or trialHitMarker not found, nan
#                 reward = np.nan
#                 trialend = np.nan
#                 rewprob = np.nan
#         except IndexError:
#             # set values to nan in case of issues
#             reward = np.nan
#             trialend = np.nan
#             rewprob = np.nan
#         rewprobl_str = str(np.array(rewprobl, dtype = int))
        
#         # append to sess_list
#         sess_list.append([npcounter, trialstart, port, reward, trialend, sessnum,
#                           rewprob, datetime, task, event, rewprobl_str,
#                           ms_licktimes, ms_lastlick, s_licktimes])
#         npcounter += 1
    
#         if filtered_events[ind, 0] == sessionStartMarker:
#             # update session number when sessionStartMarker found
#             rewprobl = []
#             sessnum +=1
    
#         if filtered_events[ind, 0] == rewardProbMarker:
#             rewprobl.append(filtered_events[ind, 4])
    
#     # create pandas dataframe from the list of all session data
#     sessdf =  pd.DataFrame(sess_list, columns = ['trial', 'trialstart',
#                                                  'port', 'reward', 'trialend',
#                                                  'session', 'rewprob', 'eptime',
#                                                  'task', 'event', 'rewprobfull',
#                                                  'ms_licktimes', 'ms_lastlick','s_licktimes'])
    
#     # remove all lines corresponding to start of a session or reward prob marker
#     sessdf = sessdf[(sessdf.event!=sessionStartMarker) & (sessdf.event!=rewardProbMarker)].reset_index(drop = True).drop(columns = 'event')
    
#     # remove all lines that have (trialend-trialstart)> 5500 or negative (sum of tone + wait time)
#     # is this check required?
    
#     # reset trial numbers
#     sessdf['trial'] = np.arange(len(sessdf))
    
#     # add animal information
#     sessdf['animal'] = animal
    
#     # add datetime
#     sessdf['datetime'] = pd.to_datetime(sessdf.eptime, unit='s')

#     return sessdf

# IN CURRENT USE - 2023-10-26
def trializer_v3(df, list_sessionMarker,
                 trialStartMarker, trialHitMarker,
                 trialMissMarker, sessionStartMarker,
                 trialProbMarker, rewardProbMarker, animal, arms = 4):
    
    ''' Function to convert a df of joined .dat files with column names in order :
    'event', 'value', 'sm', 'tp', 'smtime', 'eptime', 'dw', 'ww'
    
    INPUTS: df, list of session state machine numbers, trial start marker, trial rewarded marker,
    trial miss/unrewarded marker, session start marker, trial probability marker, session reward prob marker, 
    animal name, num arms in task
    
    OUTPUT: single trial-wise dataframe of one animal
    2023-10-26
    '''
    # initialize var
    sessnum = 0
    npcounter = 0
    sess_list = []
    rewprobl = []

    event_lookup = {
        trialHitMarker: 1,
        trialMissMarker: 0,
    }
    list_events = [trialStartMarker, trialHitMarker, trialMissMarker, trialProbMarker, sessionStartMarker,
                   trialProbMarker, rewardProbMarker]

    # filter events as np array to increase speed and avoid repetitive indexing
    filtered_events = df[(df['sm'].isin(list_sessionMarker)) & (df['event'].isin(list_events))].to_numpy()
    indices = np.where((filtered_events[:, 0] == trialStartMarker) | (filtered_events[:, 0]==sessionStartMarker) | (filtered_events[:, 0]==rewardProbMarker))[0]

    for ind in indices:
        # extract information from the current trialStartMarker
        trialstart = filtered_events[ind, 4]
        port = filtered_events[ind, 1]
        datetime = filtered_events[ind, 5]
        task = filtered_events[ind, 2]
        event = filtered_events[ind, 0]

        # select events occurring after trialStartMarker
        nextEvents = filtered_events[ind:]

        try:
            # find the index of trialMissMarker or trialHitMarker in nextEvents
            status_indices = np.where((nextEvents[:, 0] == trialMissMarker) | (nextEvents[:, 0] == trialHitMarker))[0]
            statusIndex = status_indices[0] if status_indices.size > 0 else None

            # find the index of trialProbMarker in nextEvents
            rewprob_index = np.where(nextEvents[:, 0] == trialProbMarker)[0][0] 
            #contLine: if np.any(nextEvents[:, 0] == trialProbMarker) else None

            if statusIndex is not None:
                # find reward, trialend, and rewprob based on identified events
                reward = event_lookup[nextEvents[statusIndex, 0]]
                trialend = nextEvents[statusIndex, 4]
                rewprob = nextEvents[rewprob_index, 1] if rewprob_index is not None else np.nan
            else:
                # if trialMissMarker or trialHitMarker not found, nan
                reward = np.nan
                trialend = np.nan
                rewprob = np.nan
        except IndexError:
            # set values to nan in case of issues
            reward = np.nan
            trialend = np.nan
            rewprob = np.nan

        rewprobl_str = str(np.array(rewprobl, dtype = int))
        # append to sess_list
        sess_list.append([npcounter, trialstart, port, reward, trialend, sessnum, rewprob, datetime, task, event, rewprobl_str])
        npcounter += 1

        if filtered_events[ind, 0] == sessionStartMarker:
            # update session number when sessionStartMarker found
            rewprobl = []
            sessnum +=1

        if filtered_events[ind, 0] == rewardProbMarker:
            rewprobl.append(filtered_events[ind, 4])

    # create pandas dataframe from the list of all session data
    sessdf =  pd.DataFrame(sess_list, columns = ['trial', 'trialstart',
                                   'port', 'reward', 'trialend',
                                   'session', 'rewprob', 'eptime', 'task', 'event', 'rewprobfull'])

    # remove all lines corresponding to start of a session or reward prob marker
    sessdf = sessdf[(sessdf.event!=sessionStartMarker) & (sessdf.event!=rewardProbMarker)].reset_index(drop = True).drop(columns = 'event')

    # remove all lines that have (trialend-trialstart)> 5500 or negative (sum of tone + wait time)
    # is this check required?

    # reset trial numbers
    sessdf['trial'] = np.arange(len(sessdf))

    # add animal information
    sessdf['animal'] = animal
    
    # add datetime
    sessdf['datetime'] = pd.to_datetime(sessdf.eptime, unit='s')

    return sessdf

# OLD - 2023-10-26
def trializer_v2(df, stateMachines, trialStartMarker, trialHitMarker, trialMissMarker, sessionStartMarker, trialProbMarker):
    ''' Makes a trialwise df of .dat files, 
        Input = df of .dat, List of stateMachines to be analyzed, trialStartMarker, trialHitMarker, trialMissMarker, sessionStartMarker
    Output = trialwise df
    '''
    lateSess = df[df[2].isin(stateMachines)]
    lateSessNPtimes = lateSess[lateSess[0] == trialStartMarker][4]
    lateSessRewtimes = lateSess[lateSess[0] == trialHitMarker][4]
    lateSessMisstimes = lateSess[lateSess[0] == trialMissMarker][4]
    
    pd.options.mode.chained_assignment = None
    lateSessdf = pd.DataFrame()
    lateSessdf['trial#'] = np.arange(0, len(lateSessNPtimes))
    lateSessdf['trialstart'] = lateSessNPtimes.values
    lateSessdf['port'] = lateSess[lateSess[0] == trialStartMarker][1].values
    lateSessdf['reward'] = np.nan
    lateSessdf['trialend'] = np.nan
    lateSessdf['session#'] = np.nan
    lateSessdf['rewprob'] = np.nan
    lateSessdf['datetime'] = pd.to_datetime(lateSess[lateSess[0] == trialStartMarker][5], unit='s')
    sessioncounter = 0
    npcounter = 0
    
    event_lookup = {
        trialHitMarker: 1,
        trialMissMarker: 0,
    }
    
    for ind, event in enumerate(lateSess[0]):
        if event == trialStartMarker:
            nextEvents = lateSess.iloc[ind:]
            try:
                statusIndex = min(np.where(nextEvents[0].isin([trialMissMarker, trialHitMarker]))[0])
                rewprob = min(np.where(nextEvents[0] == trialProbMarker)[0])
                lateSessdf.loc[npcounter, 'reward'] = event_lookup[nextEvents.iloc[statusIndex][0]]
                lateSessdf.loc[npcounter, 'trialend'] = nextEvents.iloc[statusIndex, 4]
                lateSessdf.loc[npcounter, 'rewprob'] = nextEvents.iloc[rewprob, 1]
            except ValueError:
                lateSessdf.loc[npcounter, 'reward'] = np.nan
                lateSessdf.loc[npcounter, 'trialend'] = np.nan
                lateSessdf.loc[npcounter, 'rewprob'] = np.nan
            npcounter += 1
        elif event == sessionStartMarker:
            lateSessdf.loc[npcounter:, 'session#'] = sessioncounter
            sessioncounter += 1
    
    pd.options.mode.chained_assignment = 'warn'
    return lateSessdf

def rew_prob_extractor(df, arms, rewardProbMarker):
    ''' Extracts reward prob as dict from given df for set number of bandit arms.
    Input: df of .dat file, number of arms, rewardProbMarker
    Output: rewardProb dict
    '''
    # extract all rew prob as dict 
    rewardProb = {}
    session_count = 0
    regret = []
    temp = []

    for ind, prob in enumerate(df[df[0]==rewardProbMarker][4]):
        temp.append(prob)
        rewardProb[session_count] = temp

        if (ind+1)%arms==0:
            session_count+=1
            temp = []
    return rewardProb

# currently doing this
def transition_matrixv3(sessdf, col1='port', col2 = 'choice_t1', shifter=-1, normalize = 'index'):
    pd.options.mode.chained_assignment = None
    sessdf.loc[:, col2] = sessdf.groupby(['animal', 'session#'])[col1].shift(shifter).astype('category')
    sessdf = sessdf.dropna()
    tm = pd.crosstab(sessdf[col1], sessdf[col2], normalize = normalize, dropna = False)
    pd.options.mode.chained_assignment = 'warn'
    return tm

# the mundane way of doing it
def transition_matrixv2(sessdf, grouped, col):
   
    t1, t = [], []
    for sessnum, group in grouped:
        for ind, row in group.iterrows():
            try:
                t_port = sessdf[col][ind]
                t1_port=sessdf[col][ind+1]
                t.append(t_port)
                t1.append(t1_port)            
            except KeyError as e:
                pass

    # for all i, j, compute: 
    # how many times does it go from i to j 

    prob_array=np.zeros((arms,arms))

    for p in range(len(t1)):
        for i in range(arms):
            for j in range(arms):
                temp = np.zeros((arms, arms))
                if ((t[p]==i+1) & (t1[p]==j+1)):
                    temp[i,j]=1
                    prob_array[i,j]+= temp[i,j]
    return prob_array

# CALCULATES REWARD RATE FOR THE WHOLE DICTIONARY
def calc_rr(sessdf, trialsinsess=150, arg = 'head', numtrials = 150):
    rr = {}
    for ind, (animal, group) in enumerate(sessdf.groupby('animal')):
        filtered = group.groupby('session#').filter(lambda x: x.reward.size >= trialsinsess)
        if arg == 'head':
            rr[animal] = np.mean(filtered.groupby(['session#']).head(numtrials)['reward'])
        elif arg == 'tail':
            rr[animal] = np.mean(filtered.groupby(['session#']).tail(numtrials)['reward'])
    rr = sort_dict(rr)
    return rr