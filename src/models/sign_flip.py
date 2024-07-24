import numpy as np
from sklearn.covariance import LedoitWolf, EmpiricalCovariance
from src.data.fname_conventions import get_src_ortho_fname, get_time_embedded_subject, get_sign_corrected_fname
from pathlib import Path as Path


def pcorr_from_precision_and_diags(C, d1, d2):
    if np.any(np.isnan(C)):
        raise ValueError("Input matrix contains NaNs.")
    if np.any(d1 < 0) or np.any(d2 < 0):
        raise ValueError("Cannot have negative entries in autoprecision")
    d1[d1 > 1e-14] = 1 / d1[d1> 1e-14]
    d2[d2 > 1e-14] = 1 / d2[d2> 1e-14]
    return -np.diag(d1) @ C @ np.diag(d2)

def pcorr_from_precision(C):
    if np.any(np.isnan(C)):
        raise ValueError("Input matrix contains NaNs.")
    d_C = np.copy(np.diag(C))
    if np.any(d_C < 0):
        raise ValueError("Precision matrix cannot have negative values in the diagonal, but found {}".format(d_C))
    valid_entries = np.abs(d_C) > 1e-14
    d_C[valid_entries] = 1/(d_C[valid_entries])
    
    if np.any(np.isnan(d_C)):
        raise ValueError("Diagonal inversion restricted to non zero entries yielded a NaN")
    diag = np.diag(np.sqrt(d_C))
    return -diag @ C @ diag

def split_lagged_global_by_lag(C, n_lags, order="D"):
    if np.any(np.isnan(C)):
        raise ValueError("NaNs in input covariance")
    if np.any(np.diag(C) < 0):
        raise ValueError("Variance of a variable cannot be negative.")
    d = int(C.shape[1] / n_lags)
    covars = np.zeros((d,d,n_lags))
    mid_lag = (n_lags-1)//2
    
    lagged_diags = np.zeros((d, n_lags))
    if order=="D":
        mid_lag_col = mid_lag*d
        for l in range(n_lags):
            for i in range(d):
                for j in range(d):
                    covars[i,j,l] = C[l*d+i,mid_lag_col + j]
            lagged_diags[:, l] = np.diag(C[l*d:(l+1)*d,l*d:(l+1)*d])
    elif order=="L":
        for l in range(n_lags):
            for i in range(d):
                for j in range(d):
                    covars[i,j,l] = C[n_lags*i+l,n_lags*j + mid_lag]
    else:
        raise NotImplementedError("Unknown ordering. Columns are either lag contiguous or dimension contiguous (e.g -L, -L+1, ...,L or D1, D2, ..., Dk")
    return covars, lagged_diags

def compute_pcorr_multi_lags(X, n_lags, order, cov_mode):
    if np.any(np.isnan(X)):
        raise ValueError("Input matrix contains NaNs, please fix it.")
    covar_lags = None
    if cov_mode == "empirical":
        covar_lags = EmpiricalCovariance(store_precision=True).fit(X).covariance_
    elif cov_mode == "ledoit":
        covar_lags = LedoitWolf(store_precision=True).fit(X).covariance_
    else:
        raise NotImplementedError("Unknown covariance mode, should be empirical or ledoit")
    # Return as covariances by lags
    autoprecision = np.diag(covar_lags)
    covars_split, lagged_diags = split_lagged_global_by_lag(covar_lags, n_lags, order)
    # For each covar, compute the inverse and transform to valid partial correlation
    L = (n_lags -1)//2
    for l in range(n_lags):
        covars_split[:,:,l] = pcorr_from_precision_and_diags(np.linalg.inv(covars_split[:,:,l]), lagged_diags[:,l], lagged_diags[:, L])
    return covars_split

def compute_pcorr_matrices(subjects_fif_dir, subject_name_list, L, mmapFile, cov_mode="empirical"):
    # load first subject with time-embedding
    # load first subject to get dimensionality
    data_subject = np.load(get_time_embedded_subject(subjects_fif_dir, subject_name_list[0], L))[L:-L]
    print("Loaded base subject")
    n_lags = 2*L + 1;
    d = int(data_subject.shape[1]//n_lags)
    n_subjects = len(subject_name_list)    
    # Compute the dimension and allocate array (might be a memmap later)
    pcorr_mats = np.zeros((d,d,n_subjects, n_lags))
    pcorr_mats[:,:,0,:] = compute_pcorr_multi_lags(data_subject, n_lags, order="D", cov_mode=cov_mode)
    for i in range(1,n_subjects):
        print("Loading {}".format(subject_name_list[i]))
        data_subject = np.load(get_time_embedded_subject(subjects_fif_dir, subject_name_list[i], L))[L:-L]
        pcorr_mats[:,:,i,:] = compute_pcorr_multi_lags(data_subject, n_lags, order="D", cov_mode=cov_mode)
        print("Pcorr done for this subject.")
    return pcorr_mats

def abs_sum_diff(a,b):
    return abs(a-b) - abs(a+b)

def gain_cost(flips, p_corr_mats):
    cost = 0
    D = p_corr_mats.shape[0]
    S = p_corr_mats.shape[2]
    L = p_corr_mats.shape[3]
    
    if len(p_corr_mats.shape) < 4:
        raise ValueError("pcorr matrices should be D x D x N_subjects x N_lags, but were {}".format(p_corr_mats.shape))
    if S == 0:
        raise ValueError("partial correlation matrices do not have enough subjects! Third dimension should be > 1")
    if flips.shape[0] != S or flips.shape[1] != D:
        raise ValueError("flips matrix should be N_subjects times N_dimensions !")
    
    if np.any(np.isnan(p_corr_mats)):
        raise ValueError("Input partial correlation contains NaNs, please correct the matrix first")
    for i in range(D):
        for j in range(D):
            for l in range(L):
                c_sum = 0
                for s in range(S):
                    c_sum += flips[s,i]*flips[s,j]*p_corr_mats[i,j,s,l]
                cost += abs(c_sum)
    return cost / S
    
def flip_diff(flips, p_corr_mats, i0, s0):
    cost = 0;
    D = p_corr_mats.shape[0]
    S = p_corr_mats.shape[2]
    L = p_corr_mats.shape[3]
    
    if flips.shape[0] != S or flips.shape[1] != D:
        raise ValueError("flips matrix should be N_subjects times N_dimensions !")
    
    for j in range(D):
        if j != i0:
            for l in range(L):
                a_i0j = 0
                a_ji0 = 0
                b_i0j = flips[s0, i0] * flips[s0,j] * p_corr_mats[i0,j,s0,l]
                b_ji0 = flips[s0, i0] * flips[s0,j] * p_corr_mats[j,i0,s0,l]
                for s in range(S):
                    if s != s0:
                        a_i0j += flips[s, i0] * flips[s, j] * p_corr_mats[i0,j,s,l]
                        a_ji0 += flips[s, i0] * flips[s, j] * p_corr_mats[j,i0,s,l]
                cost += abs_sum_diff(a_i0j, b_i0j) + abs_sum_diff(a_ji0, b_ji0)
    return cost / S

def suggest_n_random_flips(flips, n_flips, p_corr_mats):
    # Generate random participant and dimension.
    N = flips.shape[0]
    D = flips.shape[1]
    flips_target = np.copy(flips)
    participants_random_sequence = np.random.choice(N, n_flips)
    dimensions_random_sequence = np.random.choice(D, n_flips)
    for n in range(n_flips):
        i0 = dimensions_random_sequence[n]
        s0 = participants_random_sequence[n]
        gain_diff = flip_diff(flips_target, p_corr_mats, i0, s0)
        if gain_diff > 0:
            # Flip accepted
            flips_target[s0,i0] *= -1
    return flips_target

def suggest_flips(n_restarts, n_iters, p_corr_mats):
    """
    Suggest flips which will maximize gain across participants and lags on partial correlation agreement.
    The algorithm starts by randomly initialization of the flip signs.
    Then it randomly selects participant and entry to flip, and keep the flip if it leads to increase of gain function.
    After n_iters such attempts at flips, the algorithm stops.
    It is restarted n_restarts times, with the best solution of flips returned at the end (ie the one with highest gain)
    """
    best_flips = None
    best_gain = -np.inf
    D = p_corr_mats.shape[0]
    S = p_corr_mats.shape[2]
    for r in range(n_restarts):
        # Initialize the flip matrix with random signs and start search
        flip_start = np.random.choice([-1,1], (S, D))
        flips = suggest_n_random_flips(flip_start, n_iters, p_corr_mats)
        # Compute total gain
        gain_c = gain_cost(flips, p_corr_mats)
        
        # Check that the gain is non-negative. It is a sum of absolute values and should thus be non negative.
        assert gain_c >= 0, "Gain should be non negative, was {} for flips {}".format(gain_c, flips)
        if gain_c > best_gain:
            best_gain = gain_c
            best_flips = flips
    return best_flips, best_gain


def compute_flips_participants_and_apply(subjects_fif_dir, subject_names, n_restarts, n_iters, L, cov_mode):
    pcorr_mats = compute_pcorr_matrices(subjects_fif_dir, subject_names, L, "", cov_mode=cov_mode)
    print("Starting flip suggestions")
    flips, gain = suggest_flips(n_restarts, n_iters, pcorr_mats)
    print("Done.")
    # Save both
    print("Saving partial correlations and flipping vectors...")
    np.save(Path(subjects_fif_dir, "pcorr_mats.npy"), pcorr_mats)
    np.save(Path(subjects_fif_dir, "flips.npy"), flips)
    print("Done.")
    # Reload data of the subjects
    for i, s in enumerate(subject_names):
        # Apply sign correction and save resulting data
        print("Starting sign correction of {}".format(s))
        X = np.load(get_src_ortho_fname(subjects_fif_dir, s))
        X = np.diag(flips[i,:]) @ X
        np.save(get_sign_corrected_fname(subjects_fif_dir, s), X)
        print("Saved.")
    
    