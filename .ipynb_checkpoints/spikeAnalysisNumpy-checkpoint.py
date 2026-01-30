# do everything with numpy:
# convert to microvolts using conversion factor, use only first 10^6 samples, otherwise too large
# uv_recording = ((recording.get_traces()[0:100000,:64])*channel_gain).T
uv_recording = ((ephy_recording.get_traces(end_frame = 60000, channel_ids = working_channels))*channel_gain[:64]).T

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