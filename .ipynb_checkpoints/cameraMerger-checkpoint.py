from opconNosepokeFunctions import *
from supplementaryFunctions import *
import glob

import os
import pandas as pd
import numpy as np

def get_files(log_folder='L:/Box1/', extension='.dat'):
    """
    Retrieves all files with the specified extension from the provided folder(s).
    """
    if isinstance(log_folder, list):
        return [f for fol in log_folder for f in os.listdir(os.path.realpath(fol)) if f.endswith(extension)]
    return [f for f in os.listdir(os.path.realpath(log_folder)) if f.endswith(extension)]

def gen_camera_df(event_files):
    # Prepare lists to collect data
    data = []
    cam_frame_start = []

    for filename in event_files:
        # Read events and frames files
        events = pd.read_csv(filename, names=['pinstatus', 'camtime', 'pitime'], skiprows=1)
        with open(filename.replace('events', 'frames')) as f:
            cam_frame_start.append(f.readline().strip())
        events['filename'] = os.path.basename(filename)
        data.append(events)

    # Concatenate all event data into a single DataFrame
    camera_df = pd.concat(data, ignore_index=True)

    # Add additional columns
    camera_df['rec_time'] = camera_df['filename'].str[4:9].astype(int)
    camera_df['human_time'] = pd.to_datetime(camera_df['pitime'], unit='s')
    camera_df['vid_start'] = np.nan
    camera_df['human_date'] = camera_df['human_time'].dt.date
    camera_df.loc[camera_df.groupby(['human_date', 'filename']).head(1).index, 'vid_start'] = cam_frame_start
    camera_df['vid_start'] = camera_df['vid_start'].ffill()

    return camera_df

def trializer_v4(df, list_sessionMarker,
                 trialStartMarker, trialHitMarker,
                 trialMissMarker, sessionStartMarker,
                 trialProbMarker, rewardProbMarker,
                 lickMarker, animal, camera_df = None, arms = 4):
    
    ''' Function to convert a df of joined .dat files with column names in order :
    'event', 'value', 'sm', 'tp', 'smtime', 'eptime', 'dw', 'ww', 'gpio_lick'
    
    INPUTS: df, list of session state machine numbers, trial start marker, trial rewarded marker,
    trial miss/unrewarded marker, session start marker, trial probability marker, session reward prob marker, 
    lick marker, animal name, num arms in task
    
    OUTPUT: single trial-wise dataframe of any number of animals
    2024-12-01
    '''
    if camera_df is not None:
        # merge dfs for camera information
        df['datetime'] = pd.to_datetime(df.eptime, unit='s')
        tempdf = pd.merge_asof(
        left = df[(df.event == lickMarker) & (df.value == 1)],
        right = camera_df[camera_df.pinstatus == 1],
        left_on = 'datetime',
        right_on = 'human_time',
        direction = 'nearest',
        tolerance = pd.Timedelta('1sec')
        )
        final_df = df.merge(
            tempdf,
            how='left',
            on=df.columns.tolist()
        )
    else: return "camera_df not supplied"

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
                   trialProbMarker, rewardProbMarker, lickMarker]
    
    # filter events as np array to increase speed and avoid repetitive indexing
    filtered_events = final_df[(final_df['sm'].isin(list_sessionMarker)) & (final_df['event'].isin(list_events))].to_numpy()
    indices = np.where((filtered_events[:, 0] == trialStartMarker) | (filtered_events[:, 0] == sessionStartMarker) |
                       (filtered_events[:, 0] == rewardProbMarker))[0]
    
    for ind in indices:
        # extract information from the current trialStartMarker
        trialstart = filtered_events[ind, 4]
        port = filtered_events[ind, 1]
        dttime = filtered_events[ind, 5]
        task = filtered_events[ind, 2]
        event = filtered_events[ind, 0]
    
        # select events occurring after trialStartMarker
        nextEvents = filtered_events[ind:]
    
        # try:
        # find the index of trialMissMarker or trialHitMarker in nextEvents
        status_indices = np.where((nextEvents[:, 0] == trialMissMarker) | (nextEvents[:, 0] == trialHitMarker))[0]
        lick_indices = np.where(((nextEvents[:, 0]) == lickMarker) & (nextEvents[:, 1] == 1) & (nextEvents[:, 8] == 1))[0]
        statusIndex = status_indices[0] if status_indices.size > 0 else None
        lickIndex = lick_indices[lick_indices<status_indices[1]] if status_indices.size > 1 else np.array([])
    
        if (lickIndex.size > 1):
            ms_licktimes = str(nextEvents[lickIndex][:, 4]).strip('[]')
            ms_lastlick = nextEvents[lickIndex][:, 4][-1]
            s_licktimes = str(nextEvents[lickIndex][:, 5]).strip('[]')
        
            f = nextEvents[lickIndex][:, 17]
            vid_folder = np.unique(f[~pd.isnull(f)])[0] if f[~pd.isnull(f)].size > 0 else 0
            vid_filename = str(nextEvents[lickIndex][:, 13]).strip('[]')
            poke_frame_idx = nextEvents[lickIndex][:, 18][0]
            # lick_frame_idx = str(nextEvents[lickIndex][:, 18][1:]).strip('[]')
            lick_frame_idx = np.nan
        else:
            ms_licktimes, ms_lastlick, s_licktimes, vid_folder, vid_filename = np.nan, np.nan, np.nan, np.nan, np.nan
            poke_frame_idx, lick_frame_idx = np.nan, np.nan
            
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
            # ms_lastlick = np.nan
                    
        # except IndexError as e:
        #     # set values to nan in case of issues
        #     reward = np.nan
        #     trialend = np.nan
        #     rewprob = np.nan
            
        rewprobl_str = str(np.array(rewprobl, dtype = int))
        
        # append to sess_list
        sess_list.append([npcounter, trialstart, port, reward, trialend, sessnum,
                          rewprob, dttime, task, event, rewprobl_str,
                          ms_licktimes, ms_lastlick, s_licktimes,
                          vid_filename, vid_folder, poke_frame_idx, lick_frame_idx])
        npcounter += 1
        # print(reward)
    
        if filtered_events[ind, 0] == sessionStartMarker:
            # update session number when sessionStartMarker found
            rewprobl = []
            sessnum +=1
    
        if filtered_events[ind, 0] == rewardProbMarker:
            rewprobl.append(filtered_events[ind, 4])
    
    # create pandas dataframe from the list of all session data
    sessdf =  pd.DataFrame(sess_list, columns = ['trial', 'trialstart',
                                                 'port', 'reward', 'trialend',
                                                 'session', 'rewprob', 'eptime',
                                                 'task', 'event', 'rewprobfull',
                                                 'ms_licktimes', 'ms_lastlick', 's_licktimes',
                                                 'vid_filename', 'vid_folder', 'poke_frame_idx', 'lick_frame_idx'])
    
    # # remove all lines corresponding to start of a session or reward prob marker or lick!!
    sessdf = sessdf[~(sessdf.event.isin([sessionStartMarker, rewardProbMarker, lickMarker]))].reset_index(drop = True).drop(columns = 'event')
    
    # # remove all lines that have (trialend-trialstart)> 5500 or negative (sum of tone + wait time)
    # # is this check required?
    
    # # reset trial numbers
    sessdf['trial'] = np.arange(len(sessdf))
    
    # # add animal information
    sessdf['animal'] = animal
    
    # # add datetime
    sessdf['datetime'] = pd.to_datetime(sessdf.eptime, unit='s')

    return sessdf


animal = 'Vulpix'
box = 8
date = '2024-08-05'

# path to the animal data
path = glob.glob(rf'L:\4portProb/*/{animal}')[0]

files = exclude_files(get_files(path))
files = [file for file in files if (file >=f'{animal}-{date}*.dat')&(file <=f'{animal}-2024-08-14*.dat')]
files.sort()
list_sessionMarker = [13]
toneStartMarker = 23
trialStartMarker = 81
trialHitMarker = 51
trialMissMarker = 86
trialUnrewMarker = 87
sessionStartMarker = 61
rewardProbMarker = 83
trialProbMarker = 88
lickMarker = 22

set_cwd(path)
df = merge_files_to_df(files)

## MAKE CAMERA DF
# automating and adding camera df info to dat files
tic()
# get all animals 
current_animals = ['Vulpix']

# before date 
before_date = '20240814'

# after date
after_date = '20240810'

# get files
for animal in current_animals:
    
    # path to the animal data
    path = glob.glob(rf'L:\4portProb/*/{animal}')[0]

    # folders of video
    folders = np.array(sorted(glob.glob(rf'{path}\Video/**/')))

    # select after date event files only
    if after_date is not None:
        folders = folders[folders > f'{path}\\Video\\{after_date}\\']

    # select before date event files only
    if before_date is not None:
        folders = folders[folders < f'{path}\\Video\\{before_date}\\']

    # get event, frames files (assuming they exist here!!)
    event_files = [
        file for folder in folders for file in sorted(glob.glob(f'{folder}*.events'))
    ]
    frames_files = [
        file for folder in folders for file in sorted(glob.glob(f'{folder}*.frames'))
    ]

    # get information for a set of dates 
    data = []
    camera_df = gen_camera_df(event_files)
        
    # get frame ids for each event in the dataframe
    frame_idx = []
    for i in range(len(frames_files)):
    
        # load frames file
        frames = np.loadtxt(frames_files[i])
    
        # turn all frames into camera time
        frames = frames[0] + frames[1:]
    
        # get events filename to match
        event_fname = frames_files[i].rstrip('frames')+'events'
        date_folname = pd.to_datetime(os.path.split(os.path.split(event_fname)[0])[1]).date()
        
        # search for nearest frame for each event 
        events_in_vid = camera_df[(camera_df['filename']==os.path.basename(event_fname)
                                  ) & (camera_df['human_date']==date_folname)].camtime
        frame_idx.append([np.where(frames <= i)[0].max() for i in events_in_vid])
            
    # add information to df by flattening the frame_idx list
    camera_df['frame_idx'] = [idx for l_idx in frame_idx for idx in l_idx]

toc()

# MAKE SESSDF with camera df
sessdf = trializer_v4(df, list_sessionMarker,
                      trialStartMarker, trialHitMarker,
                      trialMissMarker, sessionStartMarker,
                      trialProbMarker, rewardProbMarker,
                      lickMarker, animal, camera_df)