# banditAnalysis
Updates from previous repository:
update the processed dataset and remove duplicates after reading into memory
breaking down jupyter notebook into single scripts for better control 
# bandit analysis notebooks and code base
trying to do better version control here

## Paper figures

Each plot has a short script in [figs/](figs/), built on helpers in [utils/](utils/). Data paths and analysis settings (expert cutoff, trials per block, smoothing, mean or median across rats) are in [utils/config.py](utils/config.py), the rats of each cohort in [utils/cohorts.py](utils/cohorts.py). [docs/data_columns.md](docs/data_columns.md) describes the trial tables, the cleaning and every measure.

| Plot | Script |
|---|---|
| 1–2 Reward rate: every block; block 1 → block 2 | [figs/reward_rate.py](figs/reward_rate.py) |
| 3 Switch rate: every block; block 1 → block 2 | [figs/switch_rate.py](figs/switch_rate.py) |
| 4 Switch distance: every block; block 1 → block 2 | [figs/switch_distance.py](figs/switch_distance.py) |
| 5 Transition matrices, and rewarded − unrewarded | [figs/transition_matrices.py](figs/transition_matrices.py) |
| 6 P(switch) after rewarded and unrewarded trials, block 1 | [figs/switch_by_reward.py](figs/switch_by_reward.py) |
| 7 P(switch to each distance) after rewarded and unrewarded trials, block 1 | [figs/switch_by_distance.py](figs/switch_by_distance.py) |
| 8 Switch distance after rewarded and unrewarded trials, block 1 | [figs/switch_distance_by_reward.py](figs/switch_distance_by_reward.py) |
| P(switch to each distance) whatever the outcome, block 1 | [figs/switch_by_distance.py](figs/switch_by_distance.py) |
| Reward rate, switch rate and switch distance for weeks 1–8, block 1; sessions and rats per week | [figs/weekwise.py](figs/weekwise.py) |
| Blocked vs unblocked and cued vs uncued rats, first block and every block | [figs/group_comparisons.py](figs/group_comparisons.py) |

Create the environment with `conda env create -f environment-paper.yml`. From the repository root, run one script with `python -m figs.reward_rate`, or all of them with `python -m figs.make_all` (the data are read once). The scripts also run cell by cell in VS Code. Figures (.svg, .png) go to `figs/output/` and the numbers behind each figure (.csv) to `figs/intermediate/`.
