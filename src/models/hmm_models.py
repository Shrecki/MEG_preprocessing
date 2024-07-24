from hmmlearn import hmm as hmm
from src.data.fname_conventions import get_pca_transformed_time_embedding
import numpy as np
import mne

def fit_and_predict_proba_subjects(subjects_fif_dir, subject_names, n_states, n_components, is_flipped, L):
    hmm_data = hmm.GaussianHMM(n_components=n_states, implementation="log")
    X = None
    lengths = []
    # Load data
    for s in subject_names:
        x_s = np.load(get_pca_transformed_time_embedding(subjects_fif_dir, s, L=L, is_flipped=is_flipped, n_comps=n_components))
        lengths.append(x_s.shape[0])
        if X is None:
            X = x_s
        else:
            X = np.concatenate([X, x_s])
    # Fit data
    hmm_data.fit(X, lengths)
    
    # Predict data
    state_probas = hmm_data.predict_proba(X, lengths)
    
    # Return model, predicted data and lengths
    return hmm_data, state_probas, lengths

def create_raw_states_data(predicted_probas, ref_data):
    n_states = predicted_probas.shape[1]
    state_ch_names = ["State {}".format(i+1) for i in range(n_states)]
    info = mne.create_info(ch_names=state_ch_names, sfreq=250, ch_types=["bio"]*n_states)
    raw = mne.io.RawArray(predicted_probas.T, info = info, first_samp = ref_data.first_samp)
    raw.set_meas_date(ref_data.info["meas_date"])
    raw.set_annotations(ref_data.annotations)
    return raw