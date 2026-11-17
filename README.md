# banditAnalysis
## Repository for behavioural analysis of rat data in a four-armed bandit task
Rishika Sharma, Hanuma S. Billakurthi, Ashesh K. Dhawale

## Analysis style
Most figures/script are derivatives of a largely functional programming style. Clearly not a trained programmer.

## Paper figures

Each plot has a short script in [figs/](figs/), built on helpers in [utils/](utils/). Data paths and analysis settings (expert cutoff, trials per block, smoothing, mean or median across rats) are in [utils/config.py](utils/config.py), the rats of each cohort in [utils/cohorts.py](utils/cohorts.py). [docs/data_columns.md](docs/data_columns.md) describes the trial tables, the cleaning and every measure.

| Plot | Script |
|---|---|
| 1–2 Reward rate: every block; block 1 → block 2 | [figs/reward_rate.py](figs/reward_rate.py) |
| 3 Switch rate: every block; block 1 → block 2; block 1 vs block 2 per rat (trials 1–100) | [figs/switch_rate.py](figs/switch_rate.py) |
| 4 Switch distance: every block; block 1 → block 2; block 1 vs block 2 per rat (trials 1–100) | [figs/switch_distance.py](figs/switch_distance.py) |
| 5 Transition matrices, and rewarded − unrewarded | [figs/transition_matrices.py](figs/transition_matrices.py) |
| 6 P(switch) after rewarded and unrewarded trials, block 1 | [figs/switch_by_reward.py](figs/switch_by_reward.py) |
| 7 P(switch to each distance) after rewarded and unrewarded trials, block 1 | [figs/switch_by_distance.py](figs/switch_by_distance.py) |
| 8 Switch distance after rewarded and unrewarded trials, block 1 | [figs/switch_distance_by_reward.py](figs/switch_distance_by_reward.py) |
| P(switch to each distance) whatever the outcome, block 1 | [figs/switch_by_distance.py](figs/switch_by_distance.py) |
| Reward rate, switch rate and switch distance for weeks 1–8, block 1; sessions and rats per week | [figs/weekwise.py](figs/weekwise.py) |
| Choice raster of one rat's sessions (rat and sessions set at the top; optional slider), and one example session per cohort with 3 home blocks and 1 away block coloured by environment; not run by make_all | [figs/raster_generator.py](figs/raster_generator.py) |
| Example reward profiles (reward % at ports 1–4): the 4 structured and 4 of the 32 shuffled profiles | [figs/reward_profiles.py](figs/reward_profiles.py) |
| Schematic: share of structured and shuffled blocks in the correlated and uncorrelated environments (alluvial plot, shares set at the top) | [figs/env_proportions.py](figs/env_proportions.py) |
| Sessions per week per rat (UU, UB, SB): each of weeks 1–8, and pooled for weeks 1–8 and expert data; total expert sessions per rat | [figs/sessions_per_week.py](figs/sessions_per_week.py) |
| Blocked vs unblocked and cued vs uncued rats, first block and every block | [figs/group_comparisons.py](figs/group_comparisons.py) |

Create the environment with `conda env create -f environment-paper.yml`. From the repository root, run one script with `python -m figs.reward_rate`, or all of them with `python -m figs.make_all` (the data are read once). The scripts also run cell by cell in VS Code. Figures (.svg, .png) go to `figs/output/` and the numbers behind each figure (.csv) to `figs/intermediate/`.
