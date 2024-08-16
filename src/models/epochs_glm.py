import numpy as np
import pandas as pd
from nilearn.glm.first_level import run_glm
import mne
from src.models.hmm_models import create_raw_states_data
from src.data.fname_conventions import get_subject_raw_path, get_icaed_annotated_fname
from src.data.event_extractor import get_events_from_annotated_raw


def concatenate_epochs_as_single_mat(epochs):
    data_trans = np.vstack(np.swapaxes(epochs.get_data(), 1, 2))
    d = epochs.get_data()
    for i in range(d.shape[0]):
        assert np.all(data_trans[i*351:(i+1)*351] == d[i].T)
    return data_trans

def create_design_matrix_epoch_wise(conditions, epochs, event_dict):
    design_matrix = np.zeros((len(epochs), len(conditions) + 1))
    design_matrix[:,0] = 1 # Intercept
    event_id_to_dict = { event_dict[k]: k for k in event_dict.keys()}

    for i in range(len(epochs)):
        # Get the event to condition ID
        ev = epochs[i].events[0,2]
        #event_seq[i] = ev
        event = event_id_to_dict[ev].split('_')[0]
        design_matrix[i, conditions.index(event) + 1] = 1
    # Lastly, append one thing: a constant offset
    des_mat = pd.DataFrame(design_matrix, columns=["intercept"] + conditions)
    return des_mat


def create_design_mat_and_run_glm(raw, events, event_dict, tmin, tmax, baseline, conditions):
    epochs = mne.Epochs(raw, events=events,tmin=tmin, tmax=tmax, baseline=baseline)
    epochs_data = epochs.get_data() # n_channels x n_epochs x n_times
    print(epochs_data.shape)
    n_chans = epochs_data.shape[1]
    n_times = epochs_data.shape[2]
    # Create the design matrix across epochs. This is easy
    design_mat = create_design_matrix_epoch_wise(conditions, epochs, event_dict)
    
    coeffs = []
    for chan in range(n_chans):
        t_s_content = []
        for t in range(n_times):
            # Run GLM for the channel and time in the epochs
            dat = epochs_data[:,chan,t]
            labels, results = run_glm(dat.reshape((-1,1)),design_mat.to_numpy(), verbose=10)
            t_s_content.append(results)
        coeffs.append(t_s_content)
    return design_mat, coeffs

# From data:
# For each dimension:
#    For each timepoint in an epoch:
#        get data[d, t,:] <- take across all epochs
#        get corresponding events (which simply correspond to each epoch, in fact so we can recover these once)
#        construct design matrix (again: because epoch structure is conserved can be done exactly once)
#        fit glm
#        store results

def extract_intercept_coeffs(coeffs):
    fitted_state_courses = np.zeros((len(coeffs), len(coeffs[0]), 7))
    n_times = len(coeffs[0])
    for i in range(len(coeffs)):
        for t in range(n_times):
            # Get the regression result and extract first parameter / coefficient.
            c = coeffs[i][t]
            reg_res = list(c.values())[0]
            beta = reg_res.theta
            fitted_state_courses[i,t, :] = beta.flatten()
    return fitted_state_courses

def run_two_level_glm(subjects_filtered, subjects_fif_dir, predicted_probas, cum_lengths, conditions, tmin, tmax, baseline, n_lags=None):
    # For each subject, store N_channels x N_times coefficients
    n_regressors = len(conditions) + 1
    n_subjects = len(subjects_filtered)
    n_times = 351
    n_channels = 6
    population_coefficients = np.zeros((n_subjects, n_channels, n_times, n_regressors))

    # Run a first level GLM for each subject

    for i, s in enumerate(subjects_filtered):
        data_subject = mne.io.read_raw(get_icaed_annotated_fname(get_subject_raw_path(subjects_fif_dir, s)))
        if n_lags is not None:
            times = data_subject.times
            data_subject.crop(tmin=times[n_lags],tmax=times[-n_lags-1])
        events, events_dict = get_events_from_annotated_raw(data_subject)
        raw_i = create_raw_states_data(predicted_probas[cum_lengths[i]:cum_lengths[i+1],:], data_subject)
        # First level
        design_mat, coeffs = create_design_mat_and_run_glm(raw_i, events,events_dict, tmin, tmax, baseline, conditions)

        # Extract intercept coeffs
        population_coefficients[i,:,:] = extract_intercept_coeffs(coeffs)
    # Run second level 
    population_thetas = np.zeros((n_channels, n_times, n_regressors))
    for c in range(n_channels):
        for t in range(n_times):
            for theta_i in range(n_regressors):
                labels, results = run_glm(population_coefficients[:,c,t,theta_i].reshape((-1,1)),np.ones((n_subjects,1)), verbose=10)
                res = list(results.values())[0]
                population_thetas[c, t, theta_i] = res.theta.flatten()
    return population_coefficients, population_thetas
