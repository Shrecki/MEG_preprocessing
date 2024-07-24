from sklearn.decomposition import IncrementalPCA
from src.data.fname_conventions import get_src_ortho_fname, get_time_embedded_subject, get_pca_transformed_time_embedding, get_sign_corrected_fname
from numpy.lib.stride_tricks import sliding_window_view
import numpy as  np

def compute_time_embedding(X, L):
    n_samples = X.shape[0]
    n_dims = X.shape[1]
    strides_points = n_dims*L
    y=np.pad(X, ((L,L),(0,0)), mode='reflect')
    time_embedding = sliding_window_view(y.flatten(), (2*L+1)*n_dims)[::n_dims]
    return time_embedding


def time_embedding_step(subjects_fif_dir, subject_name, L, is_flipped=False):
    # Load subject data after orthogonalisation
    
    X = None
    # IF the subject was flipped
    if is_flipped:
        X = np.load(get_sign_corrected_fname(subjects_fif_dir, subject_name)).T
    else:
        # If the subject was not flipped, it merely got out of the 
        X = np.load(get_src_ortho_fname(subjects_fif_dir, subject_name)).T
    # Perform time embedding
    embedded = compute_time_embedding(X, L)
    # Save
    np.save(get_time_embedded_subject(subjects_fif_dir, subject_name, L, is_flipped), embedded)

def group_embedding_and_pca(subjects, subjects_fif_dir, whiten, L, is_flipped, n_components):
    # Get first subject and perform its temporal embedding to get batch size
    # Load time embedding
    X = np.load(get_time_embedded_subject(subjects_fif_dir, subjects[0], L, is_flipped))
    group_pca = IncrementalPCA(n_components=n_components, whiten=whiten, batch_size=X.shape[0])
    
    print("Starting incremental PCA...")
    group_pca.partial_fit(X)
    print("OK for {}".format(subjects[0]))
    for s in subjects[1:]:
        X = np.load(get_time_embedded_subject(subjects_fif_dir, s, L, is_flipped))
        group_pca.partial_fit(X) # Add to group pca
        print("OK for {}".format(s))
    print("Fitting done, projecting.")
    # Now that group PCA is here, we should project the data
    for s in subjects:
        # load temporal embedding back
        X = np.load(get_time_embedded_subject(subjects_fif_dir, s, L, is_flipped))
        # save result
        X_t = group_pca.transform(X)
        print("{} projected...".format(s))
        np.save(get_pca_transformed_time_embedding(subjects_fif_dir, s, L, is_flipped, n_components), X_t)
        print("And saved.")
        