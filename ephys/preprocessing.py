import numpy as np
import spikeinterface as si
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
from probeinterface import ProbeGroup, generate_tetrode


import os
os.chdir('/mnt/pve/Homes/rishika/ephysData')
session_fol = 'Bayleef_2025-01-29_11-03-00_006'
data_path = os.path.expanduser(f'~/ephysData/Bayleef/Ephys/{session_fol}')

recording = si.extractors.read_openephys(data_path)

# select ephys channels only
selection = ['CH'+str(i) for i in np.arange(1,65)]
ephys_recording = recording.select_channels(selection)

# make a channel map 
channelmapoe = [40, 38, 36, 34,
                48, 46, 44, 42,
                56, 54, 52, 50,
                58, 64, 62, 60,
                63, 61, 59, 57,
                55, 53, 51, 49,
                47, 45, 43, 41,
                39, 37, 35, 33,
                25, 27, 29, 31,
                17, 19, 21, 23,
                9, 11, 13, 15,
                1, 3, 5, 7,
                4, 6, 8, 2,
                10, 12, 14, 16,
                18, 20, 22, 24,
                26, 28, 30, 32]

# subtract 1 to fit python indexing
channelmappythonic = np.array(channelmapoe) - 1

# create probegroup and set channel locations for sorting purposes
probegroup = ProbeGroup()
for i in range(16): # we have 16 tetrodes
    tetrode = generate_tetrode()
    tetrode.move([i * 300, 0]) # make lots of space between tetrodes
    probegroup.add_probe(tetrode)

probegroup.set_global_device_channel_indices(channelmappythonic) # tetrodes arranged based on channelmap
ephys_recording = ephys_recording.set_probegroup(probegroup, group_mode='by_probe')


print('---------------created tetrode groups---------------------')

bad_channel_ids, info = spre.detect_bad_channels(ephys_recording)

ephys_recording = ephys_recording.remove_channels(bad_channel_ids)

print('---------------removed bad channels-----------------------')
split_recording_dict = ephys_recording.split_by("group")

preprocessed_recording = []
sortings = {}

for grouper, chan_rec in split_recording_dict.items():

	filtered_recording = spre.bandpass_filter(chan_rec, freq_min = 300, freq_max = 6000)

	referenced_recording = spre.common_reference(filtered_recording, operator = 'median')

	preprocessed_recording.append(referenced_recording)

	sorting = ss.run_sorter(
		sorter_name ='mountainsort5',
		recording = referenced_recording,
		folder = f'{session_fol}/sorter_output/{grouper}',
		filter = False,
		verbose = True,
        n_jobs = 8
		)

	sorting.save(folder = f'{session_fol}/sorted/{grouper}')
	print(f'----------------------sorted tetrode {grouper} ---------------------')

	analyzer = si.create_sorting_analyzer(
		sorting=sorting,
		recording=referenced_recording,
		format="zarr",
		folder=f"{session_fol}/analyzer_output/{grouper}")
	print('----------------------created analyzer---------------------')

	# which is equivalent to this:
	job_kwargs = dict(n_jobs=8, chunk_duration="1s", progress_bar=True)
	compute_dict = {
	    'random_spikes': {'method': 'uniform', 'max_spikes_per_unit': 500, 'save':True},
	    'waveforms': {'ms_before': 1.0, 'ms_after': 2.0, 'save':True},
	    'templates': {'operators': ["average", "median", "std"], 'save':True}
	}
	analyzer.compute(compute_dict, **job_kwargs)

	print('----------------------computed features---------------------')