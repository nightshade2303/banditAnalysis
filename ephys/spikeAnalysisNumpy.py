import numpy as np
import glob
import spikeinterface.extractors as se
from probeinterface import ProbeGroup, generate_tetrode

animal= 'Onix'
folder_name = "Onix_2025-07-21_11-56-51"
data_path = fr'L:\4portProb_ephys\Box1_ephys\{animal}\Ephys\{folder_name}'
recording = se.read_openephys(data_path, stream_name = 'Record Node 103#OE_FPGA_Acquisition_Board-100.Rhythm Data')
# events = se.read_openephys_event(data_path)
print(recording)
# with open(glob.glob(data_path+'/**/structure.oebin', recursive = True)[0], 'r') as f:
#     oebin = json.loads(f.read())

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

import spikeinterface.widgets as sw

# do everything with numpy:
# convert to microvolts using conversion factor, use only first 10^6 samples, otherwise too large
# uv_recording = ((recording.get_traces()[0:100000,:64])*channel_gain).T
uv_recording = ((ephys_recording.get_traces(end_frame = 60000, channel_ids = working_channels))*channel_gain[:64]).T

# Define the sampling rate and frequencies
samprate = 30000
f1 = 300  # HalfPowerFrequency1 in Hz
f2 = 6000  # HalfPowerFrequency2 in Hz
order = 4  # FilterOrder

# Design the Butterworth bandpass filter
b, a = signal.butter(order, [f1, f2], btype='band', fs = 30000)

filtered = signal.filtfilt(b, a, uv_recording)

# plot voltage trace
fig = plt.figure()
pn = 1
for ch in working_channels:
    ax = plt.subplot(8,8,pn)
    ax.set_title(f'ch {ch}')
    ax.plot(uv_recording[ch], label='Recorded', linewidth = 0.5)
    pn+=1
plt.suptitle('Filtered traces')
fig.supxlabel('Samples')
fig.supylabel(r'$\mu V$')
plt.tight_layout()

# remove common mode noise (median subtraction?)
cmnoise = np.median([filtered[ch] for ch in working_channels], axis = 0)
med_subtracted = filtered - cmnoise

# plot median subtracted voltage trace
fig = plt.figure(figsize = (20, 10))
pn = 1

for ch in working_channels:
    ax = plt.subplot(8, 8, pn)
    ax.set_title(f'ch {ch}')
    ax.plot(med_subtracted[ch], label='Recorded', linewidth = 0.5)
    ax.set_ylim(150, -150)
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    pn+=1

plt.suptitle('Median subtracted traces')
fig.supxlabel('Samples')
fig.supylabel(r'$\mu V$')
plt.tight_layout()

# plot histogram of voltage signal at each channel
import seaborn as sns
fig = plt.figure(figsize = (20, 10))
pn = 1
for ch in working_channels:
    ax = plt.subplot(8,8,pn)
    sns.histplot(med_subtracted[ch], stat = 'probability')
    ax.set_title(f'ch {ch}')
    pn+=1
plt.suptitle('Voltage distribution')
fig.supylabel('Probability')
fig.supxlabel(r'$\mu V$')
plt.tight_layout()