"""Make every paper figure in one Python session, so the data are read once: python -m figs.make_all"""
import runpy

import matplotlib

matplotlib.use('Agg')  # save the figures without opening windows
import matplotlib.pyplot as plt  # noqa: E402

SCRIPTS = ['reward_rate', 'switch_rate', 'switch_distance', 'transition_matrices',
           'switch_by_reward', 'switch_by_distance', 'switch_distance_by_reward', 'weekwise', 'group_comparisons']

for name in SCRIPTS:
    print(f'figs/{name}.py')
    runpy.run_module(f'figs.{name}', run_name='__main__')
    plt.close('all')
