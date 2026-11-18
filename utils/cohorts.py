"""Rats in each cohort (block_an_dict and an_dict in multiAnimalAnalysis.ipynb)."""

# Rat lists as in the MATLAB analysis (unblocked_rats, blocked_rats, str_rats).

# uncorrelated environment with block changes (UB; task 'unstr' in sessdf), cued or uncued block changes
CUED = ['Emolga', 'Giratina', 'Ivysaur', 'Jigglypuff', 'Lugia']
UNCUED = ['Bidoof', 'Quaxly', 'Sableye', 'Torchic', 'Uxie', 'Vanillish', 'Whismur', 'Yamper', 'Zorua']

# correlated environment (SB; task 'str' in hsb_data); HSB_F_006-011 were implanted, 012-014 were not
CORRELATED = ['HSB_F_006', 'HSB_F_007', 'HSB_F_008', 'HSB_F_009', 'HSB_F_010',
              'HSB_F_011', 'HSB_F_012', 'HSB_F_013', 'HSB_F_014']

# rats of each cohort, keyed by the task the cohort was trained on
COHORTS = {'unstr': CUED + UNCUED, 'str': CORRELATED}

# uncorrelated environment without block changes (UU; task 'unstr' in sessdf): most sessions are one block
UNBLOCKED = ['test05022023', 'Blissey', 'Chikorita', 'Darkrai', 'Eevee', 'Goldeen', 'Hoppip', 'Inkay', 'Jirachi',
             'Kirlia', 'Mesprit', 'Nidorina', 'Oddish', 'Quilava', 'Raltz', 'Shinx', 'Umbreon', 'Wartortle',
             'Cresselia', 'Haxorus']

# groups compared within the uncorrelated environment
GROUPS = {'blocked': CUED + UNCUED, 'unblocked': UNBLOCKED, 'cued': CUED, 'uncued': UNCUED}
