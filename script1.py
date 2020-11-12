# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 11:42:50 2020

@author: P70049035
"""
import os
import numpy as np
import mne
from mne.preprocessing import (ICA, create_eog_epochs, create_eog_epochs, corrmap)

vhdr_name = 'RAW/Pilots/Pilot4_H/Pilot4_H_Iso.vhdr'
decim = 2.0
#print(type(decim))

#eog = 'Fp1', 'Fp2', 'AF7', 'AF8' #electrodes for eye movements, default ('HEOGL', 'HEOGR', 'VEOGb')
#misc = #analox channels for auxiliary signals, default 'auto'

raw = mne.io.read_raw_brainvision(vhdr_name, preload = True)

print(raw.info)
print(raw.info.keys())
print(raw.info['ch_names'])
l1 = raw.info['ch_names']
if 'Fp2' not in l1:
    print('Fp2 is not in the list')
else: 
    print('Fp2 is in the list')


print(len(raw.info['ch_names'])) #132
#print(raw.info['nchan'])#132
#print(raw.info['subject_info']) #none
#print(raw.info['events']) #[]

#raw.plot() #duration = 10, n_channels = 30
#raw.plot_psd(fmax=50) #power spectral density, fmax = 50, plot frequ below 50 i.e.
#%% exclude/drop channels I dont use, audio etc
#excl = ['Dunno', 'Audio', 'FSR', 'AudioLeft', 'AudioRight', 'Flex']
#incl = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'AF8', 'AF4', 'F2', 'FCz', 'F9', 'AFF1h', 'FFC1h', 'FFC5h', 'FTT7h', 'FCC3h', 'CCP1h', 'CCP5h', 'TPP7h', 'P9', 'PPO9h', 'PO9', 'O9', 'OI1h', 'PPO1h', 'CPP3h', 'CPP4h', 'PPO2h', 'OI2h', 'O10', 'PO10', 'PPO10h', 'P10', 'TPP8h', 'CCP6h', 'CCP2h', 'FCC4h', 'FTT8h', 'FFC6h', 'FFC2h', 'AFF2h', 'F10', 'AFp1', 'AFF5h', 'FFT9h', 'FFT7h', 'FFC3h', 'FCC1h', 'FCC5h', 'FTT9h', 'TTP7h', 'CCP3h', 'CPP1h', 'CPP5h', 'TPP9h', 'POO9h', 'PPO5h', 'POO1', 'POO2', 'PPO6h', 'POO10h', 'TPP10h', 'CPP6h', 'CPP2h', 'CCP4h', 'TTP8h', 'FTT10h', 'FCC6h', 'FCC2h', 'FFC4h', 'FFT8h', 'FFT10h', 'AFF6h']
#excl_chan = mne.pick_channels(ch_names = raw.info['ch_names'], include = incl, exclude = excl)
#print(excl_chan)#126
#raw.plot(n_channels = len(excl_chan)) #not possible to plot numpy.ndarray

excl_chan = mne.pick_types(raw.info, meg = False, eeg = True, misc = False)
print(excl_chan)#127
#eeg_indices = mne.pick_types(raw.info, meg = False, eeg = True, misc = False) #drop channels from raw object
#eeg_data, times = raw[eeg_indices]
#print(mne.pick_info(raw.info, eeg_indices)['nchan'])
raw.drop_channels(['Dunno', 'Audio', 'FSR', 'AudioLeft', 'AudioRight', 'Flex']) #drop channels from raw data
print(raw.info)
#raw.plot()
##raw.plot(duration = 60, n_channels = len(excl_chan1), remove_dc = False)

#%%line noise
# dont see line noise
fig = raw.plot_psd(tmax=np.inf, fmax=250, average=True)
# add some arrows at 60 Hz and its harmonics:
for ax in fig.axes[:2]:
    freqs = ax.lines[-1].get_xdata()
    psds = ax.lines[-1].get_ydata()
    for freq in (50, 100, 150, 200):
        idx = np.searchsorted(freqs, freq)
        ax.arrow(x=freqs[idx], y=psds[idx] + 18, dx=0, dy=-12, color='red',
                 width=0.1, head_width=3, length_includes_head=True)

#%% downsample 250Hz
#Since downsampling reduces the timing precision of events, we recommend first extracting epochs and downsampling the Epochs object
#data_res = mne.filter.resample(raw, down = decim, npad = 'auto') #downsample, not working!!!!
raw.resample(250, npad = 'auto')
print(raw.info)
#raw.plot()

#%% Delete Data before first event and after last event - not necessary?

#%% set montage
#raw.set_montage('standard_1005')

#%% High-pass filter
raw.filter(0.5,45, fir_design = 'firwin') #0.1
#raw.plot()
print(raw.info)
#%% Clean raw data - noisy channels, interpolate
#in tutorial they say that I shall pick them by hand, mark as bad and then exclude and interpolate them
# https://mne.tools/0.15/auto_tutorials/plot_artifacts_correction_rejection.html
# https://mne.tools/stable/auto_tutorials/preprocessing/plot_15_handling_bad_channels.html#sphx-glr-auto-tutorials-preprocessing-plot-15-handling-bad-channels-py

#%% re-reference
#raw.set_eeg_reference(ref_channels = ['FCz'])
raw_new_ref = mne.add_reference_channels(raw, ref_channels =['Fp2']) #add the online reference electrode back to the data as flat channel = Fp2 is the REFERNCE!
raw_new_ref.set_eeg_reference(ref_channels =['TP9', 'TP10']) #average of 'mastoid'electrodes
#raw_new_ref.plot() #plots

print(raw_new_ref.info)
#%% Overview of artifact detection
# low frequ drifts
# ocular artifacts
# https://mne.tools/stable/auto_tutorials/preprocessing/plot_40_artifact_correction_ica.html#tut-artifact-ica

#%% ica
#default = fastica, newer and more robust = picard
ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
ica.fit(raw_new_ref)
#%%
ica.plot_sources(raw_new_ref) #we can use the unfiltered data here (https://mne.tools/stable/auto_tutorials/preprocessing/plot_40_artifact_correction_ica.html#tut-artifact-ica)
#%%
#ten_twenty_montage = mne.channels.make_standard_montage('standard_1005')
#raw.set_montage('standard_1005')
#print(ten_twenty_montage)

#ica.plot_components() #pass instance and get interactive graphs, does not WORK, because the montage we have (actiCap, easy cap 128) is not uset in mne atm

#%%
ica.plot_overlay(raw_new_ref, exclude = [1, 4], picks = 'eeg') #see how data wold look like it exclude this component

#%%
# ica.plot_properties(raw_new_ref, picks=[1]) #does not work, or picks=ica.excude

#%%
ica.exclude = [1, 4]
#%%
orig_raw = raw.copy()
raw.load_data()
ica.apply(raw)

# show some frontal channels to clearly illustrate the artifact removal
chns = ['Fp1', 'Fp2', 'AF7', 'AF8']
# https://mne.tools/stable/auto_tutorials/intro/plot_10_overview.html#preprocessing
chan_idxs = [raw.ch_names.index()]
raw.plot(order = artifact_picks, n_channels=len(artifact_picks))
reconst_raw.plot(order=artifact_picks, n_channels=len(artifact_picks))




#%% Events