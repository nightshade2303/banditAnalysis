from opconNosepokeFunctions import *
from supplementaryFunctions import *
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
                   'Eevee', 'Goldeen', 'Hoppip', 'Inkay', 'Jirachi', 'Kirlia', 'Mesprit',
                   'Nidorina', 'Oddish', 'Phione', 'Quilava', 'Raltz', 'Shinx', 'Togepi',
                   'Umbreon', 'Vulpix', 'Xatu', 'Yanma', 'Zacian',
                   'Alakazam', 'Bayleef', 'Cresselia', 'Dratini']
boxes = listdir_fullpath("L:/4portProb/")

# full dataset generation (n = 21), with provision for demarcating lesioned sessions using date of reentry into task
sessdf = pd.DataFrame()
files = {}
df_dict = {}
# sess_dict = {}

for box in boxes:
    for animal in os.listdir(box): 
        if animal in current_animals:
            print(animal)
            direc = np.nonzero(np.array(os.listdir(box))==animal)[0][0]
            files[box] = exclude_files(get_files(box+'/'+os.listdir(box)[direc]+'/'))
            files[box].sort()
            
            try:
                df_dict[box] = merge_files_to_df(files[box])
                
            except pd.errors.EmptyDataError as e:
                print('empty file not read', animal) 
                
            except Exception as e:
                print(f"Unexpected {e}, {type(e)}", animal)

            animal_sessdf = trializer_v3(df_dict[box], list_sessionMarker, trialStartMarker, trialHitMarker,
                                  trialMissMarker, sessionStartMarker, trialProbMarker, rewardProbMarker, animal)
                
            sessdf = pd.concat([sessdf, animal_sessdf], ignore_index = True)
            
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
mask5 = (sessdf.datetime > datetime.datetime(2024, 9, 30)) & (sessdf.animal == 'Zacian')
sessdf.loc[mask5, 'task'] = 'dls'

# reducing filesize on disk
sessdf.to_csv('L:/4portProb_processed/sessdf.csv')
# def conv_Str(sessdf):
#     # Convert 'rewprobfull' to string once
#     rewprobfull_str = sessdf['rewprobfull'].astype(str)
    
#     # Slice the parts for all rows at once
#     part1 = rewprobfull_str.str[2:4]
#     part2 = rewprobfull_str.str[8:10]
#     part3 = rewprobfull_str.str[14:16]
#     part4 = rewprobfull_str.str[20:22]

#     return part1 + " " + part2 + " " + part3 + " " + part4

######## get trials that are structured within unstructured task ###########
# struc = ['[80 46 15 10]', '[46 80 46 15]', '[15 46 80 46]', '[10 15 46 80]', '[80 46 10 15]', '[15 10 46 80]']
# env = []
# sessdf['rewprobfull'] = sessdf.rewprobfull.apply(lambda x: x.translate({ord(i): None for i in '[] '})
#                                                                     .replace('.0', '')
#                                                                     .split(','))
# for ind, i in enumerate(sessdf.rewprobfull.apply(lambda x: np.array([eval(i) for i in x], dtype = int))):
#     if str(i) in struc:
#         env.append('str')
#     else:
#         env.append('unstr')
# sessdf['env'] = env

# rp_string = conv_Str(sessdf).to_numpy()
# sessdf['rpfull_string'] = rp_string
# exclude = list(sessdf.rpfull_string.sort_values().unique()[:6])+['70   ', '20 20 20 10', '30   ', '40   ',]
# sessdf = sessdf[~sessdf.rpfull_string.isin(exclude)]
# sessdf = sessdf[~sessdf.duplicated(subset = ['animal', 'session', 'trialstart', 'eptime'], keep = False)]
window = 7
sessdf['rr'] = sessdf.groupby(['animal', 'session'], as_index = False).reward.rolling(window, center=True).mean().reward

sessdf["animal"] = sessdf["animal"].astype("category")
sessdf["task"] = sessdf['task'].astype("category")
sessdf["rewprobfull"] = sessdf['rewprobfull'].astype(str).astype("category")


# saving
sessdf.to_csv('L:/4portProb_processed/sessdf.csv')
print('Program ended successfully.')