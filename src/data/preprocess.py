from src.utils.helpers import *
import mne
import numpy as np
from pathlib import Path as Path
from src.data.fname_conventions import *
from src.data.coreg_atlas import put_atlas_to_subject_space, create_atlas_dict
from src.data.event_extractor import *
from mne_connectivity.envelope import symmetric_orth


"""
The different steps of preprocessing are broken into individual methods, which are then called or skipped.
They are assumed to happen in a set order.

"""


def filtering_step(raw_path, overwrite):
    raw = mne.io.read_raw(raw_path, preload=True)
    # For annotation purposes
    raw.plot(block=True)
    
    raw_filt = raw.copy().notch_filter([50, 100]).filter(1, 45)
    # Save result
    raw_filt.save(get_filtered_fname(raw_path), overwrite=overwrite)
    # Clean memory
    del raw
    del raw_filt

def ica_step(raw_path, overwrite):
    raw = mne.io.read_raw(get_filtered_fname(raw_path), preload=True)
    ica = mne.preprocessing.ICA(n_components=0.99999, max_iter="auto", random_state=97, method='picard')
    filt_raw = raw.copy().filter(l_freq=1.0, h_freq=None)
    ica.fit(filt_raw)
    ica.plot_sources(filt_raw, show_scrollbars=False, block=True)
    ica.apply(raw)
    # Save result
    raw.save(get_icaed_fname(raw_path), overwrite=overwrite)
    ica.save(get_ica_model_fname(raw_path), overwrite=overwrite)
    # Clean memory
    del raw
    del ica
        
def source_setup_step(subject_name, subjects_mri_dir, overwrite):
    # Setup the source space for whole brain volumetric
    vol_src = mne.setup_volume_source_space(subject_name, 
                                subjects_dir=subjects_mri_dir, 
                                surface= Path(subjects_mri_dir, subject_name, "bem", "inner_skull.surf"))
    # Save source space
    mne.write_source_spaces(get_source_fname(subjects_mri_dir, subject_name), vol_src, overwrite=overwrite)
    print_success()
    del vol_src
    
def bem_mesh_step(subjects_mri_dir, subject_name,overwrite):
    print_step("bem mesh creation and solution")
    model = mne.make_bem_model(subject=subject_name, subjects_dir=subjects_mri_dir, ico=4, conductivity=(0.33,))  
    bem_sol = mne.make_bem_solution(model)

    # Save solutions
    try:
        mne.write_bem_surfaces(get_bem_surface_fname(subjects_mri_dir, subject_name), model, overwrite=overwrite)
        mne.write_bem_solution(get_bem_sol_fname(subjects_mri_dir, subject_name), bem_sol, overwrite=overwrite)
    except FileExistsError:
        print("BEM already exists. Set override=True to overwrite!")

    # Clean memory
    del model
    del bem_sol
    
def manual_coreg_step(subject_name, subjects_mri_dir, raw_path):
    mne.gui.coregistration(subject=subject_name, subjects_dir=subjects_mri_dir, inst= get_icaed_fname(raw_path),block=True)
    
def fwd_step(subjects_fif_dir, subjects_mri_dir, subject_name, raw_path, overwrite=True):    
    # We need fully postprocessed icaed file
    info = mne.io.read_info(get_icaed_fname(raw_path))
    info = mne.pick_info(info, mne.pick_types(info, meg=True, eeg=False, exclude=[]))
    
    # We need trans file
    fname_trans = get_trans_file_fname(subjects_fif_dir, subject_name)
    # We need src file
    src = mne.read_source_spaces(get_source_fname(subjects_mri_dir, subject_name))
    bem_sol = mne.read_bem_solution(get_bem_sol_fname(subjects_mri_dir, subject_name))
    
    # Compute forward
    fwd = mne.make_forward_solution(info, fname_trans, src, bem_sol)

    # Save fwd
    fwd.save(get_fwd_sol_fname(raw_path),overwrite=overwrite)
    
    # Clean memory
    del fwd
    del src
    del bem_sol
    del info
    
def compute_covariances_from_events_pauses(raw_file, events, downsample_factor):
    # To define the noisy segments, we have:
    # - Time before first event
    # - Big time in between event blocks
    # Leveraging all of these should allow us to pinpoint "noise covariance" to scale sensors appropriately by prewhitening.
    # Another way to look at it: take first event of each block. Consider the timing [-15 seconds, -0.2s] before event to define windows of interest
    # For the purpose of these computations, one should not reject epochs containing bad segments, as otherwise we run the risk
    # of rejecting the entire dataset.
    retained_events = events[np.hstack((np.array([0]), np.where(np.diff(events[:, 0]) > 15000/downsample_factor)[0] + 1)),:]
    epochs_noise = mne.Epochs(raw_file, events=retained_events, tmin=-15, tmax=0, reject_by_annotation=False)
    cov_noise = mne.compute_covariance(epochs_noise, method='ledoit_wolf')
    noise_cov = mne.cov.prepare_noise_cov(cov_noise, raw_file.info)
    noise_cov.plot(raw_file.info)
    
    epochs_signal = mne.Epochs(raw_file, events=events, tmin=-0.2, tmax=1.7, reject_by_annotation=False)
    data_cov = mne.compute_covariance(epochs_signal, method='ledoit_wolf')
    data_cov.plot(raw_file.info)
    
    del epochs_noise, epochs_signal
    
    return data_cov, noise_cov

def annotate_from_ica_downsample_and_save(raw_path, downsample_freq, overwrite):
    raw_post_process_icaed = mne.io.read_raw(get_icaed_fname(raw_path), preload=True)
    raw_post_process_icaed = annotate_raw_with_events_from_table(get_icaed_fname(raw_path), raw_post_process_icaed)
    downsample_factor = raw_post_process_icaed.info['sfreq']  / downsample_freq
    raw_post_processed_meg = raw_post_process_icaed.copy().pick("meg").resample(sfreq=downsample_freq).interpolate_bads(reset_bads=True)
    raw_post_processed_meg.save(get_icaed_annotated_fname(raw_path), overwrite=overwrite)
    return raw_post_processed_meg, downsample_factor


def perform_source_reconstruction(raw_path, subject_name, subjects_fif_dir, overwrite=True):
    # Let's investigate now computation of covariance (noise and signal) in the data.
    # Load the preprocessed data
    #raw_path = os.path.join(subjects_fif_dir, subject_name, "nBack_tsss_mc.fif")
    
    
    # Add task events as annotations
    #raw_post_process_icaed.plot(block=True)
    # Downsample to 250 Hz!
    downsample_freq= 250
    raw_post_processed_meg, downsample_factor = annotate_from_ica_downsample_and_save(raw_path, downsample_freq, overwrite)
    
    # Apply bandpass filter
    raw_post_processed_meg.filter(1,45,method="iir", iir_params=dict(order=5, ftype='butter', output='sos'))

    # Extract the events from the annotations, limited only to the task (ie: no bad annotation)
    # Plot for sanity check
    events_task, event_ids = get_events_from_annotated_raw(raw_post_processed_meg)
    #raw_post_processed_meg.plot(events=events_task, block=True)
    
    # Compute covariances
    data_cov, noise_cov = compute_covariances_from_events_pauses(raw_post_processed_meg, events_task, downsample_factor)
    
    # Read forward model
    fwd = mne.read_forward_solution(get_fwd_sol_fname(raw_path))
    
    # Compute source rec result
    filters = mne.beamformer.make_lcmv(raw_post_processed_meg.info, forward=fwd, data_cov=data_cov, noise_cov=noise_cov)
    stc = mne.beamformer.apply_lcmv_raw(raw_post_processed_meg, filters)
    del fwd, filters, data_cov, noise_cov, raw_post_processed_meg
    stc.save(get_src_rec_fname(subjects_fif_dir, subject_name), overwrite=overwrite)
    del stc
    
def atlasing_step(subjects_mri_dir, subjects_fif_dir, subject_name, t1_ref, atlas_path):
    # Transform the atlas to subject space
    put_atlas_to_subject_space(subject_name, subjects_mri_dir, t1_ref, atlas_path)
    # Get list of regions
    atlas_dict = create_atlas_dict(subjects_mri_dir, subject_name, atlas_path)
    # Extract subject's signal
    # Get subject stc
    stc = mne.read_source_estimate(get_src_rec_fname(subjects_fif_dir, subject_name))
    vol_src = mne.read_source_spaces(get_source_fname(subjects_mri_dir, subject_name))
    label_tcs = stc.extract_label_time_course(labels=atlas_dict, 
                                              src=vol_src)
    np.save(get_atlased_rec_fname(subjects_fif_dir, subject_name), label_tcs)
    del stc
    

def leakage_correction(subject_name, subjects_fif_dir):
    # Get atlased data
    # Perform leakage correction
    data = np.load(get_atlased_rec_fname(subjects_fif_dir, subject_name))
    ortho_solution = symmetric_orth(data)
    np.save(get_src_ortho_fname(subjects_fif_dir, subject_name), ortho_solution)
    

def preprocess_fif(subject_name, subjects_fif_dir, subjects_mri_dir, atlas_t1_ref, atlas_thresholded, skip_steps, overwrite=False):
    # The steps:
    #   notch filter 50 / 100 Hz
    #   band pass filter [0.1 - 125 Hz]
    #   ICA filtering of bad channels (manual)
    #   Head model reconstruction
    #   Coregistration (manual)
    #   Forward computation
    #   Source reconstruction on downsampled data at 250 Hz
    #   Atlasing of source rec
    #   Orthogonalization of sources
    print(subject_name)
    print(subjects_mri_dir)
    
    raw_path = Path(subjects_fif_dir, subject_name, "nBack_tsss_mc.fif")

    
    sm = StateMachine(["filtering", "ICA", "source setup", "bem_mesh", "coreg", "fwd", "source rec", "atlasing", "source ortho"])
    
    curr_step = sm.getStateName()
    if noskip_step("filtering", curr_step, skip_steps):
        filtering_wrapped = step_wrapper("filtering", filtering_step)
        filtering_wrapped(raw_path, overwrite)
        print_success()
        
    curr_step = sm.transition()
    if noskip_step("ICA", curr_step, skip_steps):
        ica_wrapped = step_wrapper("ICA", ica_step)
        ica_wrapped(raw_path, overwrite)
        print_success()
        
    curr_step = sm.transition()
    if noskip_step("source setup", curr_step, skip_steps):
        source_setup_wp = step_wrapper("source setup", source_setup_step)
        source_setup_wp(subject_name, subjects_mri_dir, overwrite)
        print_success()
        
    curr_step = sm.transition()
    
    if noskip_step("bem_mesh", curr_step, skip_steps):
        bem_mesh_wp = step_wrapper("bem mesh creation and solution", bem_mesh_step)
        bem_mesh_wp(subjects_mri_dir, subject_name,overwrite)
        print_success()
        
    curr_step = sm.transition()
    if noskip_step("coreg", curr_step, skip_steps):
        coreg_wp = step_wrapper("manual coregistration", manual_coreg_step)
        coreg_wp(subject_name, subjects_mri_dir, raw_path)
        print_success()
        
    curr_step = sm.transition()
    if noskip_step("fwd", curr_step, skip_steps):
        fwd_wp = step_wrapper("forward model computation", fwd_step)
        fwd_wp(subjects_fif_dir, subjects_mri_dir, subject_name, raw_path,overwrite)
        print_success()
        
    curr_step = sm.transition()
    if noskip_step("source rec", curr_step, skip_steps):
        # Let's go source rec
        src_wp = step_wrapper("source reconstruction", perform_source_reconstruction)
        src_wp(raw_path, subject_name, subjects_fif_dir, overwrite=overwrite)
        print_success()
    
    curr_step = sm.transition()
    if noskip_step("atlasing", curr_step, skip_steps):
        # Let's do atlasing
        atlas_wp = step_wrapper("atlasing", atlasing_step)
        atlas_wp(subjects_mri_dir, subjects_fif_dir, subject_name, atlas_t1_ref, atlas_thresholded)
        print_success()
    curr_step = sm.transition()
    if noskip_step("source ortho", curr_step, skip_steps):
        leakage_wp = step_wrapper("Source orthogonolization", leakage_correction)
        leakage_wp(subject_name, subjects_fif_dir)
        print_success()
    print("Done with computations.")
        
