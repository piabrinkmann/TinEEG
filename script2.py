# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 16:39:26 2020

@author: P70049035
"""

"""
#Pipeline for Pilot 9
"""
from time import time

import os
import numpy as np
import mne
from mne.preprocessing import (ICA, create_eog_epochs, create_eog_epochs, corrmap)
from mne.viz import plot_events, plot_compare_evokeds
from mne import combine_evoked

path = 'C:/Users/p70049035/OneDrive/University/PhD/PhD_Studienstiftung/EEGTin_Study/Piloting_EEG/'
vhdr_name = path + 'RAW/Pilots/Pilot9_J/Pilot_9_J_iso.vhdr'
vmrk_name = path + 'RAW/Pilots/Pilot9_J/Pilot_9_J_iso.vmrk'

raw = mne.io.read_raw_brainvision(vhdr_name, preload = True)

# Read in the event information as MNE annotations
annot = mne.read_annotations(vmrk_name)
# Add the annotations to our raw object so we can use them with the data
raw.set_annotations(annot)

print(raw.info)
#print(raw.info.keys())
#print(raw.info['ch_names'])

#%% add ref channel

print(raw.info['ch_names'])
print(len(raw.info['ch_names']))

l1 = raw.info['ch_names']
if 'FCz' not in l1:
    print('FCz is not in the list')
else: 
    print('FCz is in the list')

raw_new_ref = mne.add_reference_channels(raw, ref_channels =['FCz']) #add the online reference electrode back to the data as flat channel
#raw_new_ref.plot()
#raw.set_montage('standard_1005', on_missing = 'ignore') #need the montage! change the on_missing parameter here

#%% montage
#put the .bvef file in the path where the script is stored
montage_path = path + 'CACS-128_NO_REF.bvef' #if use REF .bvef file then it does not work
montage = mne.channels.read_custom_montage(montage_path, head_size = 0.54) #.bvef is supported, head_size in meters (we have 54 cm)
print(raw.info)
print(raw.info['ch_names'])
#print(raw.info['chs'])
print(len(raw.info['ch_names']))
montage.plot()
raw_new_ref.set_montage(montage)

#%% change channel types
raw_new_ref.set_channel_types(mapping={'Audio': 'stim', 'AudioLeft': 'stim', 'AudioRight': 'stim'})

#%% Band-pass filter
raw_new_ref.filter(1,30, fir_design = 'firwin') #0.1
#raw.plot()
print(raw_new_ref.info)

#%% Re-reference
raw_new_ref.set_eeg_reference(ref_channels = ['TP9', 'TP10']) #average of 'mastoid'electrodes
#raw_new_ref = mne.add_reference_channels(raw, ref_channels =['Fp2']) #add the online reference electrode back to the data as flat channel = Fp2 is the REFERNCE!
#raw_new_ref.set_eeg_reference(ref_channels =['TP9', 'TP10']) #average of 'mastoid'electrodes
#raw_new_ref.plot() #plots
print(raw_new_ref.info)

#%% ICA
#default = fastica, newer and more robust = picard

#from time import time

#picks = mne.pick_types(raw.info, eeg = True)
#reject = None

#def run_ica(method, fit_params=None):
#    ica = ICA(n_components=20, method=method, fit_params=fit_params,
#              random_state=0)
#    t0 = time()
#    ica.fit(raw, picks=picks, reject=reject)
#    fit_time = time() - t0
#    title = ('ICA decomposition using %s (took %.1fs)' % (method, fit_time))
#    ica.plot_components(title=title)
#run_ica('picard')
#%% ICA

picks = mne.pick_types(raw_new_ref.info, eeg = True)
reject = None
method = 'picard'

ica = ICA(n_components=20, method=method, fit_params=None,
              random_state=0)
t0 = time()
ica.fit(raw_new_ref, picks=picks, reject=reject)
fit_time = time() - t0
title = ('ICA decomposition using %s (took %.1fs)' % (method, fit_time))
ica.plot_components(title=title)

#%%
ica.plot_overlay(raw_new_ref, exclude =[1, 9], picks = 'eeg')
#%%
ica.plot_properties(raw_new_ref, picks=[1, 9])
#%%
ica.exclude = [1, 9]
#%%
orig_raw = raw_new_ref.copy()
raw_new_ref.load_data()
ica.apply(raw_new_ref)

#%%
# show some frontal channels to clearly illustrate the artifact removal
chs = [ 'Fp1', 'Fp2', 'AFp1', 'AFp2',
       'AF7', 'AF8', 'AF3', 'AFz', 'AF4']
chan_idxs = [raw_new_ref.ch_names.index(ch) for ch in chs]
orig_raw.plot(order=chan_idxs, start=12, duration=4)
raw_new_ref.plot(order=chan_idxs, start=12, duration=4)

#raw.plot()
#reconst_raw.plot()
#del reconst_raw
#%% events
# Reconstruct the original events from our Raw object
events, event_ids = mne.events_from_annotations(raw_new_ref)
#%% plot events
fig = mne.viz.plot_events(events)
#%% artifact rejection in EEG channel - amplitude criterion

reject_criteria = dict(eeg=60e-6)       # 60 µV
                      

#%% Epoching
epochs = mne.Epochs(raw_new_ref, events, event_id=event_ids, tmin=-0.07, tmax=0.63,
                    baseline =(None, 0),
                    reject=reject_criteria, preload=True)
print(epochs)
#%%
# compute evoked response and noise covariance,and plot evoked
evoked = epochs.average ()
print(evoked)
#%%
title = 'My first ERP image'
evoked.plot(titles=dict(eeg=title), time_unit='s')
evoked.plot_topomap(times=[0.1], size=3., title=title, time_unit='s')
#%%
S1 = epochs["Stimulus/S  1"].average()
S2 = epochs["Stimulus/S  2"].average()
S3 = epochs["Stimulus/S  3"].average()
S4 = epochs["Stimulus/S  4"].average()
S5 = epochs["Stimulus/S  5"].average()
S6 = epochs["Stimulus/S  6"].average()
all_evokeds = [S1, S2, S3, S4, S5, S6]
print(all_evokeds)
#%% does NOT work!
#all_evokeds = [epochs[cond].average() for cond in sorted(event_ids.keys())]
#print(all_evokeds)

# Then, we can construct and plot an unweighted average of left vs. right
# trials this way, too:
#mne.combine_evoked(
#    all_evokeds, weights=[0.5, 0.5, 0.5, -0.5, -0.5, -0.5]).plot_joint(times=[0.1], title='All ERPs')
#%%
#NEW
conditions = ['Stimulus/S  1', 'Stimulus/S  2', 'Stimulus/S  3']
evokeds = {condition:epochs[condition].average()
           for condition in conditions}
pick = evokeds['Stimulus/S  1'].ch_names.index('Cz')
plot_compare_evokeds(evokeds, picks=pick, ylim=dict(eeg=(-5, 2)))
#%%
epochs['Stimulus/S  1'].plot_image(picks=['Cz'])
