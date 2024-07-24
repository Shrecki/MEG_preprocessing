from pathlib import Path as Path

def get_source_fname(subjects_dir, subject_name):
    return Path(subjects_dir, subject_name, '{}-src.h5'.format(subject_name))

def get_bem_sol_fname(subjects_mri_dir, subject_name):
    return Path(subjects_mri_dir, subject_name, '{}-bem_sol.fif'.format(subject_name))

def get_filtered_fname(raw_path):
    return Path(str(raw_path).replace('.fif', '_filtered.fif'))

def get_icaed_fname(raw_path):
    return Path(str(raw_path).replace('.fif', '_filtered_icaed.fif'))
    
def get_ica_model_fname(raw_path):
    return Path(str(raw_path).replace('.fif', '_ica-model.fif'))

def get_trans_file_fname(subjects_fif_dir, subject_name):
    return Path(subjects_fif_dir, subject_name, "raw_trans.fif")

def get_bem_surface_fname(subjects_mri_dir, subject_name):
    return Path(subjects_mri_dir, subject_name, '{}-surface_bem.fif'.format(subject_name))

def get_fwd_sol_fname(raw_path):
    return Path(str(raw_path).replace('.fif', '_fwd.fif'))

def get_src_rec_fname(subjects_fif_dir, subject_name):
    return Path(subjects_fif_dir, subject_name, "raw_source_rec-vl.stc")

def get_atlased_rec_fname(subjects_fif_dir, subject_name):
    return Path(subjects_fif_dir, subject_name, "raw_source_atlased.npy")

def get_atlas_coreg_fname(subjects_mri_dir, subject_name):
    return Path(subjects_mri_dir, subject_name, "mri", "atlas_coreg.mgz")

def get_src_ortho_fname(subjects_fif_dir, subject_name):
    return Path(subjects_fif_dir, subject_name, "raw_source_orthog.npy")

def get_events_fname(raw_path):
    return str(raw_path).replace('.fif', '_events.tsv')


def get_time_embedded_subject(subjects_fif_dir, subject_name, L, is_flipped=False):
    s = "-flipped-" if is_flipped else ""
    return Path(subjects_fif_dir, subject_name, "raw_source_time_embed{}-{}-lags.npy".format(s, L))

def get_sign_corrected_fname(subjects_fif_dir, subject_name):
    return Path(subjects_fif_dir, subject_name, "raw_source_orthog_sign_flipped.npy")

def get_pca_transformed_time_embedding(subjects_fif_dir, subject_name, L, is_flipped, n_comps):
    s = "-flipped-" if is_flipped else ""
    return Path(subjects_fif_dir, subject_name, "raw_source_pca_{}-components_time_embed{}-{}-lags.npy".format(n_comps,s, L))

def get_icaed_annotated_fname(raw_path):
    return Path(str(get_icaed_fname(raw_path)).replace(".fif", "_annotated.fif"))

def get_subject_raw_path(subjects_fif_dir, subject_name):
    return Path(subjects_fif_dir, subject_name, "nBack_tsss_mc.fif")

    