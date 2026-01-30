from utils.opconNosepokeFunctions import *
from utils.supplementaryFunctions import *

    
import glob
toneStartMarker = 23
trialStartMarker = 81
trialHitMarker = 51
trialMissMarker = 86
trialUnrewMarker = 87
sessionStartMarker = 61
rewardProbMarker = 83
trialProbMarker = 88
lickMarker = 25
unstrSessionMarker = 13
strSessionMarker = 12
list_sessionMarker = [unstrSessionMarker, strSessionMarker]

arms = 4
# change list to include new animals, if required # Xatu Goldeen Zacian Cresselia Mesprit
current_animals = ['Quaxly', 'Raikou', 'Sableye', 'Torchic', 'Uxie', 'Vanillish', 'Whismur', 'Xerneas', 'Yamper', 'Zorua']

# full dataset generation, with provision for demarcating lesioned sessions using date of reentry into task
files = {}
sessdf_dict = {}
tic()
for an in current_animals:
    # print bhai
    print(an)

    # get path for this animal's data
    path = glob.glob(rf'L:\4portProb/*/{an}')[0]

    # set as current wd
    set_cwd(path)

    # get .dat files and store names here
    files[an] = exclude_files(get_files(path))

    # since you got files, now sort them
    files[an].sort()
    latest_file = files[an][-5:]
    # merge all files to make one df
    try:
        df = merge_files_to_df(latest_file)
        
    except pd.errors.EmptyDataError as e:
        print('empty file not read') 
        
    except Exception as e:
        print(f"Unexpected {e}, {type(e)}")

    # got one df of .dat files. run trializer on it
    sessdf_dict[an] = trializer_v3(df, list_sessionMarker, trialStartMarker, trialHitMarker,
                                trialMissMarker, sessionStartMarker, trialProbMarker, rewardProbMarker, an)
    
sessdf = pd.concat(sessdf_dict, ignore_index = True)
sessdf.to_csv('L:/4portProb_processed/sessdf_daily.csv', index = False)
