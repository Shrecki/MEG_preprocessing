import os
from progress.bar import Bar
from scipy.io import loadmat
subjects_fif_dir = "/media/miplab-nas2/Data/guibert/nBack_complete/working_dir/nBack_Share_HC/MEGDATA/"
subjects_mri_dir = "/media/miplab-nas2/Data/guibert/nBack_complete/working_dir/nBack_Share_HC/MRIDATA"
healthy_subjects_codes = loadmat("/media/miplab-nas2/Data/guibert/nBack_complete/nBack_Share_HC/subjects_code_HC.mat")['HC']
with Bar("Processing...") as bar:
    for e in healthy_subjects_codes:
        p = os.path.join(subjects_fif_dir, e)
        if os.path.exists(p) and not os.path.exists(os.path.join(subjects_mri_dir, e, "surf", "rh.pial")):
            cmd = "recon-all -sd {} -subjid {} -i {} -all".format(subjects_mri_dir, e, os.path.join(p, "MEG_" + e + "_MRI_T1.nii"))
            os.system(cmd)
            bar.next()
            
with Bar("Processing...") as bar:
    for e in healthy_subjects_codes:
        p = os.path.join(subjects_fif_dir, e)
        if os.path.exists(p) and os.path.exists(os.path.join(subjects_mri_dir, e, "surf", "rh.pial")):
            cmd = "mne watershed_bem -d {} -s {}".format(subjects_mri_dir, e)
            os.system(cmd)
            bar.next()
