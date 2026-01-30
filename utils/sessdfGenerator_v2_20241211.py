from opconNosepokeFunctions import *
from supplementaryFunctions import *
import glob
<<<<<<< HEAD
=======
import datetime

>>>>>>> 80714cd (initial commit)
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
current_animals = ['test05022023', 'Blissey', 'Chikorita', 'Darkrai',
<<<<<<< HEAD
                   'Eevee', 'Goldeen', 'Hoppip', 'Inkay', 'Jirachi', 'Kirlia', 'Mesprit',
                   'Nidorina', 'Oddish', 'Phione', 'Quilava', 'Raltz', 'Shinx', 'Togepi',
                   'Umbreon', 'Vulpix', 'Xatu', 'Yanma', 'Zacian',
                   'Alakazam', 'Bayleef', 'Cresselia',
                   'Emolga','Giratina', 'Haxorus', 'Ivysaur', 'Jigglypuff', 'Lugia',
                   'Ninetales', 'Onix']
=======
                    'Eevee', 'Goldeen', 'Hoppip', 'Inkay', 'Jirachi', 'Kirlia', 'Mesprit',
                    'Nidorina', 'Oddish', 'Phione', 'Quilava', 'Raltz', 'Shinx', 'Togepi',
                    'Umbreon', 'Vulpix', 'Xatu', 'Yanma', 'Zacian',
                    'Alakazam', 'Bayleef', 'Cresselia',
                    'Emolga','Giratina', 'Haxorus', 'Ivysaur', 'Jigglypuff', 'Lugia',
                    'Ninetales', 'Onix', 'Pichu',
                    'Quaxly', 'Sableye', 'Torchic',
                    'Uxie','Vanillish', 'Whismur','Xerneas', 'Yamper', 'Zorua']
>>>>>>> 80714cd (initial commit)

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

    # merge all files to make one df
    try:
        df = merge_files_to_df(files[an])
        
    except pd.errors.EmptyDataError as e:
        print('empty file not read', an) 
        
    except Exception as e:
        print(f"Unexpected {e}, {type(e)}", an)

    # got one df of .dat files. run trializer on it
    sessdf_dict[an] = trializer_v3(df, list_sessionMarker, trialStartMarker, trialHitMarker,
                                  trialMissMarker, sessionStartMarker, trialProbMarker, rewardProbMarker, an)
    
    # output of one animal is ready, now stitch with others - most efficient way of doing this is to get a 
    # dictionary and concat after the loop (i think)
sessdf = pd.concat(sessdf_dict, ignore_index = True)

# additional details for saving - lesion information
sessdf.task = sessdf.task.replace({13:'unstr', 12:'str'})
mask = (sessdf.datetime > datetime.datetime(2023, 9, 19)) & (sessdf.animal == 'Chikorita')
sessdf.loc[mask, 'task']='ds'
mask2 = (sessdf.datetime > datetime.datetime(2023, 9, 20)) & (sessdf.animal == 'Eevee')
sessdf.loc[mask2, 'task']='ds'
mask3 = (sessdf.datetime > datetime.datetime(2023, 11, 15)) & (sessdf.animal == 'Blissey')
sessdf.loc[mask3, 'task']='dls'
mask4 = (sessdf.datetime > datetime.datetime(2023, 11, 16)) & (sessdf.animal == 'Darkrai')
sessdf.loc[mask4, 'task']='dls'
mask5 = (sessdf.datetime > datetime.datetime(2023, 11, 23)) & (sessdf.animal == 'test05022023')
sessdf.loc[mask5, 'task']='sham'
mask6 = (sessdf.datetime > datetime.datetime(2023, 12, 11)) & (sessdf.animal == 'Hoppip')
sessdf.loc[mask6, 'task']='dms'
mask7 = (sessdf.datetime > datetime.datetime(2023, 12, 11)) & (sessdf.animal == 'Kirlia')
sessdf.loc[mask7, 'task']='dms'
mask8 = (sessdf.datetime > datetime.datetime(2023, 12, 26, 18, 30)) & (sessdf.animal == 'Darkrai')
sessdf.loc[mask8, 'task']='dls_str'
mask9 = (sessdf.datetime > datetime.datetime(2023, 12, 26, 18, 30)) & (sessdf.animal == 'Kirlia')
sessdf.loc[mask9, 'task']='dms_str'
mask5 = (sessdf.datetime > datetime.datetime(2023, 12, 26, 18, 30)) & (sessdf.animal == 'test05022023')
sessdf.loc[mask5, 'task']='sham_str'
mask = (sessdf.datetime > datetime.datetime(2024, 1, 6)) & (sessdf.animal == 'Blissey')
sessdf.loc[mask, 'task']='ds'
mask2 = (sessdf.datetime > datetime.datetime(2024, 1, 6)) & (sessdf.animal == 'Hoppip')
sessdf.loc[mask2, 'task']='ds'
mask = (sessdf.datetime > datetime.datetime(2024, 2, 15)) & (sessdf.animal == 'Inkay')
sessdf.loc[mask, 'task']='dls'
mask2 = (sessdf.datetime > datetime.datetime(2024, 2, 15)) & (sessdf.animal == 'Jirachi')
sessdf.loc[mask2, 'task']='sham'
mask3 = (sessdf.datetime > datetime.datetime(2024, 3, 20)) & (sessdf.animal == 'Mesprit')
sessdf.loc[mask3, 'task'] = 'dls'
mask3 = (sessdf.datetime > datetime.datetime(2024, 3, 20)) & (sessdf.animal == 'Goldeen')
sessdf.loc[mask3, 'task'] = 'sham'
mask4 = (sessdf.datetime > datetime.datetime(2024, 5, 2)) & (sessdf.animal == 'Nidorina')
sessdf.loc[mask4, 'task'] = 'dms'
mask5 = (sessdf.datetime > datetime.datetime(2024, 5, 21)) & (sessdf.animal == 'Phione')
sessdf.loc[mask5, 'task'] = 'sham'
mask5 = (sessdf.datetime > datetime.datetime(2024, 5, 21)) & (sessdf.animal == 'Quilava')
sessdf.loc[mask5, 'task'] = 'dls'
mask5 = (sessdf.datetime > datetime.datetime(2024, 8, 4)) & (sessdf.animal == 'Raltz')
sessdf.loc[mask5, 'task'] = 'dls'
mask5 = (sessdf.datetime > datetime.datetime(2024, 7, 17)) & (sessdf.animal == 'Shinx')
sessdf.loc[mask5, 'task'] = 'oe_implant'
mask5 = (sessdf.datetime > datetime.datetime(2024, 8, 10)) & (sessdf.animal == 'Togepi')
sessdf.loc[mask5, 'task'] = 'dms'
mask5 = (sessdf.datetime > datetime.datetime(2024, 8, 27)) & (sessdf.animal == 'Umbreon')
sessdf.loc[mask5, 'task'] = 'sham'
mask5 = (sessdf.datetime > datetime.datetime(2024, 9, 17)) & (sessdf.animal == 'Vulpix')
sessdf.loc[mask5, 'task'] = 'sham'
mask5 = (sessdf.datetime > datetime.datetime(2024, 9, 30)) & (sessdf.animal == 'Xatu')
sessdf.loc[mask5, 'task'] = 'dms'
mask5 = (sessdf.datetime > datetime.datetime(2025, 3, 2)) & (sessdf.animal == 'Yanma')
sessdf.loc[mask5, 'task'] = 'dms'
mask5 = (sessdf.datetime > datetime.datetime(2024, 9, 30)) & (sessdf.animal == 'Zacian')
sessdf.loc[mask5, 'task'] = 'dls'
mask5 = (sessdf.datetime > datetime.datetime(2025, 3, 3)) & (sessdf.animal == 'Alakazam')
sessdf.loc[mask5, 'task'] = 'dls'
mask5 = (sessdf.datetime > datetime.datetime(2025, 4, 29)) & (sessdf.animal == 'Emolga')
sessdf.loc[mask5, 'task'] = 'sham'
mask5 = (sessdf.datetime > datetime.datetime(2025, 4, 29)) & (sessdf.animal == 'Giratina')
sessdf.loc[mask5, 'task'] = 'dms'
mask5 = (sessdf.datetime > datetime.datetime(2025, 4, 29)) & (sessdf.animal == 'Haxorus')
sessdf.loc[mask5, 'task'] = 'sham'
mask5 = (sessdf.datetime > datetime.datetime(2025, 4, 23)) & (sessdf.animal == 'Ivysaur')
sessdf.loc[mask5, 'task'] = 'dms'
mask5 = (sessdf.datetime > datetime.datetime(2025, 4, 23)) & (sessdf.animal == 'Jigglypuff')
sessdf.loc[mask5, 'task'] = 'sham'
mask5 = (sessdf.datetime > datetime.datetime(2025, 4, 29)) & (sessdf.animal == 'Lugia')
sessdf.loc[mask5, 'task'] = 'dms'
<<<<<<< HEAD
=======
mask5 = (sessdf.datetime > datetime.datetime(2025, 6, 11)) & (sessdf.animal == 'Onix')
sessdf.loc[mask5, 'task'] = 'oe_implant'
mask5 = (sessdf.datetime > datetime.datetime(2025, 7, 31)) & (sessdf.animal == 'Pichu')
sessdf.loc[mask5, 'task'] = 'oe_implant'
mask5 = (sessdf.datetime > datetime.datetime(2025, 11, 26)) & (sessdf.animal == 'Quaxly')
sessdf.loc[mask5, 'task'] = 'dms'
mask5 = (sessdf.datetime > datetime.datetime(2025, 11, 26)) & (sessdf.animal == 'Sableye')
sessdf.loc[mask5, 'task'] = 'dms'
mask5 = (sessdf.datetime > datetime.datetime(2025, 11, 26)) & (sessdf.animal == 'Torchic')
sessdf.loc[mask5, 'task'] = 'dls'
mask5 = (sessdf.datetime > datetime.datetime(2025, 11, 27)) & (sessdf.animal == 'Uxie')
sessdf.loc[mask5, 'task'] = 'dls'
mask5 = (sessdf.datetime > datetime.datetime(2025, 11, 26)) & (sessdf.animal == 'Vanillish')
sessdf.loc[mask5, 'task'] = 'dls'
mask5 = (sessdf.datetime > datetime.datetime(2025, 11, 27)) & (sessdf.animal == 'Whismur')
sessdf.loc[mask5, 'task'] = 'dms'
mask5 = (sessdf.datetime > datetime.datetime(2025, 10, 8)) & (sessdf.animal == 'Xerneas')
sessdf.loc[mask5, 'task'] = 'oe_implant'
mask5 = (sessdf.datetime > datetime.datetime(2025, 11, 27)) & (sessdf.animal == 'Yamper')
sessdf.loc[mask5, 'task'] = 'dls'
mask5 = (sessdf.datetime > datetime.datetime(2025, 11, 27)) & (sessdf.animal == 'Zorua')
sessdf.loc[mask5, 'task'] = 'dls'
>>>>>>> 80714cd (initial commit)

sessdf.to_csv('L:/4portProb_processed/sessdf.csv')


# reducing filesize on disk
window = 7
sessdf['rr'] = sessdf.groupby(['animal', 'session'], as_index = False).reward.rolling(window, center=True).mean().reward

sessdf["animal"] = sessdf["animal"].astype("category")
sessdf["task"] = sessdf['task'].astype("category")
sessdf["rewprobfull"] = sessdf['rewprobfull'].astype(str).astype("category")


# saving
sessdf.to_csv('L:/4portProb_processed/sessdf.csv')
print('Program ended successfully.')
toc()
