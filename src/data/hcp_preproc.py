#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 15:31:40 2023

@author: guibertf
"""

"""
Simple reproduction of the MNE-HCP tutorial to reach ERF
"""
import os
import os.path as op
from os.path import join
import glob
import numpy as np
import matplotlib.pyplot as plt
import mne
import hcp
import hcp.preprocessing as preproc
import easygui
import time

from mne.viz.topomap import _find_topomap_coords

# If not done, do:
#os.system("sshfs -o follow_symlinks guibert@stiitsrv23.epfl.ch:/media/miplab-nas2/Data/guibert/HCP_MEG/ /media/miplab-nas2/")

print("Starting!")
mne.set_log_level('WARNING')

HCP_DIR = '/media/miplab-nas2/Data/guibert/HCP_MEG/'
SUBJECTS_DIR = '/media/miplab-nas2/Data/guibert/HCP_MEG/'



subjects = sorted(glob.glob(join(SUBJECTS_DIR, '*/')))[:-2]

# === FOR A GIVEN SUBJECT !
subject = subjects[0].split('/')[-2]
recording_path = join(SUBJECTS_DIR, 'derivatives')


# ==== Brain model creation ====
msg = "Do you want to conduct BEM estimation ?"
title = "Step"
choices = ["No, I have a BEM already", "Yes please"]
id_choice = choices.index(easygui.choicebox(msg, title, choices))

meg_file = op.join(SUBJECTS_DIR, 'derivatives', subject, 'raw.fif')

if id_choice == 1:
    hcp.make_mne_anatomy(
        subject,
        subjects_dir=SUBJECTS_DIR,
        hcp_path=HCP_DIR,
        recordings_path=join(SUBJECTS_DIR, 'derivatives'))
    
    os.chdir(os.path.join(SUBJECTS_DIR, subject))
    os.system(' '.join([
                'mne', 'watershed_bem',
                '-s', subject,
                '-d', SUBJECTS_DIR,
                '-o', '--atlas']))
    
    surfaces = mne.make_bem_model(
                    subject,
                    ico=4,
                    subjects_dir=SUBJECTS_DIR, conductivity=[0.3])
    bem = mne.make_bem_solution(surfaces)    


# ==== Fiducial placement for head model ====
msg = "Do you want to manually place the fiducials ?"
title = "Step"
choices = ["No, I have done so already", "Yes please"]
id_choice = choices.index(easygui.choicebox(msg, title, choices))

if id_choice == 1:
    # This part is to be done LOCALLY, not remotely on the cluster!
    # SSHFS while following symbolic links (essential for HCP daataset)
    os.system('sshfs -o follow_symlinks guibert@stiitsrv23.epfl.ch:/media/miplab-nas2/Data/guibert/HCP_MEG/ /media/miplab-nas2/')
    
    # Perform coregistration (Note that we only have the fiducials :())
    mne.gui.coregistration(subject=subject, subjects_dir=SUBJECTS_DIR, inst=meg_file)


# ==== Event extraction ====
# Go extract the events
hcp_params = dict(
    hcp_path=HCP_DIR,
    subject=subject,
    data_type='task_working_memory')

# these values are looked up from the HCP manual
tmin, tmax = -1.5, 2.5
decim = 4
#event_dict = dict(face=1, tool=2)
event_dict = {'fixation': 0, 'face 0-back':1,  'face 2-back': 2,
              'tool 0-back': 3, 'tool 2-back': 4}

baseline = (-0.5, 0)

trial_infos = list()
all_events = list()
for run_index in [0, 1]:
    hcp_params['run_index'] = run_index
    trial_info = hcp.read_trial_info(**hcp_params)
    trial_infos.append(trial_info)
    
    trial_type_code = np.zeros(trial_info['stim']['codes'][:, 4].shape[0])
    trial_type_code[(trial_info['stim']['codes'][:, 4] == 1) & (trial_info['stim']['codes'][:, 3] == 1)] = event_dict['face 0-back'] 
    trial_type_code[(trial_info['stim']['codes'][:, 4] == 2) & (trial_info['stim']['codes'][:, 3] == 1)] = event_dict['face 2-back']
    trial_type_code[(trial_info['stim']['codes'][:, 4] == 1) & (trial_info['stim']['codes'][:, 3] == 2)] = event_dict['tool 0-back']
    trial_type_code[(trial_info['stim']['codes'][:, 4] == 2) & (trial_info['stim']['codes'][:, 3] == 2)] = event_dict['tool 2-back']
    
    events = np.c_[
        trial_info['stim']['codes'][:, 6] - 1,  # time sample
        np.zeros(len(trial_info['stim']['codes'])),
        trial_type_code  # event codes
    ].astype(int)

    #events = np.c_[
    #    trial_info['stim']['codes'][:, 6] - 1,  # time sample
    #    np.zeros(len(trial_info['stim']['codes'])),
    #    trial_info['stim']['codes'][:, 3]  # event codes
    #].astype(int)

    # for some reason in the HCP data the time events may not always be unique
    unique_subset = np.nonzero(np.r_[1, np.diff(events[:, 0])])[0]
    events = events[unique_subset]  # use diff to find first unique events
    
    # Sort events by first column to order by timestep
    events = events[np.argsort(events[:,0])]
    all_events.append(events)

n_jobs = 4
# === PER SESSION ! ===
for run_index, events in zip([0, 1], all_events):
    msg = "Do you want to preprocess this run ? (run " + str(run_index + 1) + ")"
    title = "Preprocess?"
    choices = ["Yes", "Skip"]
    choice = choices.index(easygui.choicebox(msg, title, choices))
    satisfying = choice == 1
    hcp_params['run_index'] = run_index
    print("Started loading data!")
    raw = None
    tstart = 0
    tstop = 0
    percentcut_bothside = 5;
    if choice == 0:
        # ==== Preprocess LOOP (we finish the loop when satisfied with results)
        raw = hcp.read_raw(**hcp_params)
        raw.load_data()
        raw.add_events(events)
        print("Done loading data!")
        tstart= raw.times[0] + raw.times[-1]*(percentcut_bothside/100)
        tstop = raw.times[-1]- raw.times[-1]*(percentcut_bothside/100)
    while not satisfying:
        # ==== Annotation of bad data spans & bad channels ====
        raw.plot(block=True, events=events)
        
        psd_raw = raw.compute_psd(fmax=250)
        psd_raw.plot(average=True,picks="mag",exclude="bads")
        time.sleep(5)
        raw.save(join(recording_path, subject, 'raw-run' + str(run_index).zfill(2) + '.fif'), overwrite=True)
        #raw.pick_types(meg=True, eog=True, ecg=True)
        meg_picks = mne.pick_types(raw.info, meg=True)
    
        ica_satisfying = False;
        msg = "Enter start and end time of the time serie to exclude potential noise"
        title="Start/stop times"
        fieldNames = ["start", "stop"]
        times = [tstart, tstop]
        times = [float(f) for f in easygui.multenterbox(msg, title, fieldNames)]
        
        
        while not ica_satisfying:
            # ==== Decide on bandpass filter params ====
            msg = "Enter bandpass filter frequencies prior to ICA"
            title="ICA pre-Filtering"
            fieldNames = ["low freq", "high freq"]
            fieldValues = [1, 40]
            fieldValues = [float(f) for f in easygui.multenterbox(msg, title, fieldNames)]
            
            # Copy and filter data according to filter params
            raw2 = raw.copy().filter(fieldValues[0], fieldValues[1], l_trans_bandwidth=.1, h_trans_bandwidth=1., n_jobs=n_jobs,).crop(times[0], times[1])
            
            # ==== Decide on number of ICA components ===
            msg = "Number of ICA components to use"
            title="ICA comps"
            fieldName = "n"
            n_components = 0.90 
            n_components = float(easygui.enterbox(msg, title, fieldName))
            if n_components > 1:
                n_components = int(n_components)
            
            # ==== ICA computation =====
            ica = mne.preprocessing.ICA(n_components=n_components, method='picard', random_state=23)
            ica.fit(raw2, picks=meg_picks, verbose=True)
            
            # ==== Show ICA components and manually select them ===
            fig = ica.plot_components(show=True, inst=raw2)
            plt.pause(0.1)
            ica.plot_sources(raw2, block=True)

            ica.plot_overlay(raw.crop(tstart, tstop), picks=meg_picks)
            
            # ==== If unhappy about data, go back either to annotation or ICA
            msg = "Satisfied with ICA components ?"
            title = "Checkpoint"
            choices = ["Continue", "Back to annotations", "Back to filter + ICA"]
            id_choice = choices.index(easygui.choicebox(msg, title, choices))
            
            if id_choice == 0:
                # ==== Application of ICA to unfiltered data and save everything ====
                # Apply ICA to the unfiltered data and end the ICA loop
                # Store the parameters used by the filter, the number of ICA components,
                # The ICA itself
                ica_satisfying = True
                ica.save(join(recording_path, subject, 'ica-run' + str(run_index).zfill(2) + '.fif'), overwrite=True)
            elif id_choice == 1:
                # Back to annotations
                #raw = hcp.read_raw(**hcp_params)
                #raw.load_data()
                raw.plot(block=True, events=events)
                raw.save(join(recording_path, subject, 'raw-run' + str(run_index).zfill(2) + '.fif'), overwrite=True)
                #raw.pick_types(meg=True, eog=True, ecg=True)
                meg_picks = mne.pick_types(raw.info, meg=True)
        # ==== Apply ICA to data ====
        raw_icaed = ica.apply(raw)

        # ==== Filter data ====
        msg = "Enter bandpass filter frequencies for cleaned signal"
        title="ICA post-Filtering"
        fieldNames = ["low freq", "high freq"]
        fieldValues = [1, 40]
        fieldValues = [float(f) for f in easygui.multenterbox(msg, title, fieldNames)]
        raw_icaed = raw_icaed.filter(fieldValues[0], fieldValues[1], l_trans_bandwidth=.1, h_trans_bandwidth=1., n_jobs=n_jobs,)

        # ==== Apply ref correction to data ====
        preproc.apply_ref_correction(raw_icaed)
        raw_icaed.save(join(recording_path, subject, 'raw-ICAed-run' + str(run_index).zfill(2) + '.fif'), overwrite=True)
        del raw
        # ==== Breaking data into epochs ====
        raw_icaed.pick_types(meg="mag")

        layout = mne.channels.find_layout(raw_icaed.info, ch_type="mag")
        layout.names = raw_icaed.info['ch_names']

        # Epoch data
        epochs = mne.Epochs(raw_icaed, events=events,
                            event_id=event_dict, tmin=tmin, tmax=tmax,
                            reject=None)
        face_epochs = epochs["face 2-back"]
        face_evoked = face_epochs.average()
        face_evoked.plot_topo(layout=layout)
        tool_epochs = epochs["tool 2-back"]
        tool_evoked = tool_epochs.average()
        tool_evoked.plot_topo(layout=layout)
        evoked_diff = mne.combine_evoked([face_evoked ,tool_evoked], weights=[1, -1])
        evoked_diff.pick_types(meg="mag").plot_topo(layout=layout, color="r")

        face_evoked.plot(spatial_colors=True, gfp=True)
        tool_evoked.plot(spatial_colors=True, gfp=True)

        evoked = epochs.average(by_event_type=True)

        msg = "Select a channel to inspect closely"
        title="Evoked channel selection"
        fieldName = "channel"
        selected_channel = "MEG 1811"
        selected_channel = float(easygui.enterbox(msg, title, fieldName))

        mne.viz.plot_compare_evokeds(evoked,picks=selected_channel, colors={"face 2-back": 0, "tool 2-back": 1},time_unit="ms")
        # ==== Inspection of ERP quality. If good, save preprocessing parameters, data, and exit the loop. Otherwise, start back from ICA plot to select best components
        msg = "Satisfied with ERF quality ?"
        title = "Checkpoint"
        choices = ["Continue", "Back to preprocessing"]
        id_choice = choices.index(easygui.choicebox(msg, title, choices))
        satisfying = id_choice == 0
        print("Done with basic preprocessing steps, monving on to source reconstruction on non-epoched data!")
    raw_icaed = mne.io.read_raw(join(recording_path, subject, 'raw-ICAed-run' + str(run_index).zfill(2) + '.fif'))
    raw_icaed.load_data()
    raw_icaed.pick_types(meg="mag")

    raw_icaed.compute_psd(fmax=250).plot(average=True,picks="data",exclude="bads")


    layout = mne.channels.find_layout(raw_icaed.info, ch_type="mag")
    layout.names = raw_icaed.info['ch_names']
    # Epoch data
    epochs = mne.Epochs(raw_icaed, events=events,
                        event_id=event_dict, tmin=tmin, tmax=tmax,
                        reject=None)
    face_epochs = epochs["face 2-back"]
    face_evoked = face_epochs.average()
    face_evoked.plot_topo(layout=layout)
    tool_epochs = epochs["tool 2-back"]
    tool_evoked = tool_epochs.average()
    tool_evoked.plot_topo(layout=layout)
    evoked_diff = mne.combine_evoked([face_evoked ,tool_evoked], weights=[1, -1])
    evoked_diff.pick_types(meg="mag").plot_topo(layout=layout, color="r")

    face_evoked.plot(spatial_colors=True, gfp=True)
    tool_evoked.plot(spatial_colors=True, gfp=True)

    evoked = epochs.average(by_event_type=True)

    #msg = "Select a channel to inspect closely"
    #title="Evoked channel selection"
    #fieldName = "channel"
    #selected_channel = "MEG 1811"
    #selected_channel = easygui.enterbox(msg, title, fieldName)

    #mne.viz.plot_compare_evokeds(evoked,picks=selected_channel, colors=dict(face=0, tool=1),time_unit="ms")


    # ==== Compute forward model and perform source reconstruction on the "optimally reconstructed" data
    head_mri_t = mne.read_trans(op.join(join(SUBJECTS_DIR, 'derivatives'), subject, '{}-head_mri-trans.fif'.format(subject)))
    src_fsaverage = mne.setup_source_space(subject='fsaverage', subjects_dir=SUBJECTS_DIR, add_dist=False, spacing='oct6')
    src_subject = mne.morph_source_spaces(src_fsaverage, subject, subjects_dir=SUBJECTS_DIR)
    surfaces = mne.make_bem_model(subject,ico=None, subjects_dir=SUBJECTS_DIR, conductivity=[0.3])
    bem_sol = mne.make_bem_solution(surfaces)
    bem_sol['surfs'][0]['coord_frame'] = 5

    fwd = mne.make_forward_solution(epochs.info, trans=head_mri_t, bem=bem_sol, src=src_subject)
    mag_map = mne.sensitivity_map(fwd, projs=None, ch_type='mag', mode='fixed', exclude=[], verbose=None)

    data_cov= mne.compute_covariance(face_epochs, tmin=0.01, tmax=0.5, mode="empirical")
    noise_cov= mne.compute_covariance(face_epochs, tmin=tmin, tmax=0, mode="empirical") # compute from noise session, no?

    evoked = face_epochs.average()
    filters = mne.beamformers.make_lcmv(evoked.info, forward, data_cov, reg=0.05, noise_cov=noise_cov, pick_ori="max-power", weight_norm="unit-noise-gain", rank=None)

    stc = mne.beamformer.apply_lcmv(evoked, filters)
    stc.plot(mode="stat_map")
    # ==== Save it for later use by HMM model

# Done for this participant
