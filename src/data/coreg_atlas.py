from nilearn.maskers import NiftiMasker
import nibabel as nib
import os
import numpy as np
from src.data.fname_conventions import get_atlas_coreg_fname
from pathlib import Path as Path

def threshold_4D_atlas_and_save_based_on_ref(atlas_path, ref_path, save_path, thresh):
    atlas_file = nib.load(atlas_path)
    t1_file = nib.load(ref_path)
    data = atlas_file.get_fdata()
    n_regions = data.shape[-1]
    data_mask = np.zeros((data.shape[:-1]),dtype=np.uint8).flatten()

    pos_mask = data.reshape((-1,n_regions)).sum(axis=1) > thresh

    data_mask[pos_mask] = np.argmax(data.reshape((-1,n_regions))[pos_mask],axis=1) + 1
    data_mask = data_mask.reshape((data.shape[:-1]))
    atlas_new = nib.Nifti1Image(data_mask,affine=t1_file.affine,header=t1_file.header,dtype=np.uint8)
    atlas_new.to_filename(save_path)

def put_atlas_to_subject_space(subject, subjects_mri_dir, t1_ref, atlas_path):
    mri_path = Path(subjects_mri_dir, subject, "mri")
    # Get brain extracted T1 of the subject in subject space, put it in NifTI
    os.system("mri_convert {} {}".format(Path(mri_path, "antsdn.brain.mgz"), Path(mri_path, "antsdn_brain.nii.gz")))
    # Compute atlas space => subject space transform, using the T1_ref of the atlas
    os.system("flirt -in {} -ref {} -out {} -omat {}"
              .format(t1_ref,
                      Path(mri_path, "antsdn_brain.nii.gz"),
                      Path(mri_path, "T1atlas_coreg"),
                      Path(mri_path, "atlas_to_subject_mat")))
    # Apply atlas => subject transform to the atlas file
    os.system("flirt -in {} -ref {} -out {} -applyxfm -init {} -interp nearestneighbour"
              .format(atlas_path,
                      Path(mri_path, "antsdn_brain.nii.gz"),
                      Path(mri_path, "atlas_coreg.nii.gz"),
                      Path(mri_path, "atlas_to_subject_mat")))
    # Convert back resulting coregistered file to a format freesurfer understands
    os.system("mri_convert {} {}".format(Path(mri_path, "atlas_coreg.nii.gz"), Path(mri_path, "atlas_coreg.mgz")))

def create_atlas_dict(subjects_mri_dir, subject_name, atlas_path):
    # Careful to extract and convert back as int. Otherwise labels will be floats, which results in errors!
    atlas_vals = np.unique(nib.load(atlas_path).get_fdata())[1:].astype(np.uint8)
    region_dict = {}
    for i in range(atlas_vals.size):
        region_dict["PCC_" + str(i)] = atlas_vals[i]
    # Atlas filename is expected to be in *.mgz, so convert back
    return (get_atlas_coreg_fname(subjects_mri_dir, subject_name), region_dict)