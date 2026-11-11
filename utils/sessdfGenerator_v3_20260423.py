from utils.opconNosepokeFunctions import get_files, exclude_files, set_cwd, merge_files_to_df
from utils.supplementaryFunctions import tic, toc
from utils.trializer_v4 import  apply_task_rules_from_json
from utils.trializer_v4_synced import trializer_v4_synced
import numpy as np
import pandas as pd
import glob
from datetime import datetime

arms = 4
# change list to include new animals, if required # Xatu Goldeen Zacian Cresselia Mesprit
current_animals = ['test05022023', 'Blissey', 'Chikorita', 'Darkrai',
                    'Eevee', 'Goldeen', 'Hoppip', 'Inkay', 'Jirachi', 'Kirlia', 'Mesprit',
                    'Nidorina', 'Oddish', 'Phione', 'Quilava', 'Raltz', 'Shinx', 'Togepi',
                    'Umbreon', 'Vulpix', 'Xatu', 'Yanma', 'Zacian',
                    'Alakazam', 'Bayleef', 'Cresselia',
                    'Emolga','Giratina', 'Haxorus', 'Ivysaur', 'Jigglypuff', 'Lugia',
                    'Ninetales', 'Quaxly', 'Sableye', 'Torchic',
                    'Uxie','Vanillish', 'Whismur','Xerneas', 'Yamper', 'Zorua']

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
    # try:
    #     df = merge_files_to_df(files[an])
        
    # except pd.errors.EmptyDataError as e:
    #     print('empty file not read', an) 
        
    # except Exception as e:
    #     print(f"Unexpected {e}, {type(e)}", an)

    # sessdf_dict[an] = trializer_v4_synced(df, an)
toc()

sessdf = pd.concat(sessdf_dict, ignore_index=True)
today_date = datetime.today().strftime('%Y-%m-%d')
sessdf.to_csv(f'L:/4portProb_processed/sessdf-{today_date}.csv')

# additional details for saving - lesion information (JSON-driven)
rules_path = r"C:\Users\dlab\banditAnalysis\utils\task_data.json"
sessdf = apply_task_rules_from_json(sessdf, rules_path)


# reducing filesize on disk
window = 7
sessdf['rr'] = sessdf.groupby(['animal', 'session'], as_index = False).reward.rolling(window, center=True).mean().reward

# window = 5
# sessdf['entropy'] = (sessdf.groupby(['animal', 'session'], as_index = False)
#                      .port
#                      .rolling(window, center=True)
#                      .apply(lambda x: entropy(calc_prob(x), base = 2))
#                      .port)


sessdf["animal"] = sessdf["animal"].astype("category")
sessdf["task"] = sessdf['task'].astype("category")
sessdf["rewprobfull"] = sessdf['rewprobfull'].astype(str).astype("category")
sessdf.to_csv(f'L:/4portProb_processed/sessdf-{today_date}.csv')
