from nilearn.maskers import NiftiMasker
import nibabel as nib
import os
import numpy as np
from src.data.fname_conventions import get_atlas_coreg_fname
from pathlib import Path as Path
import ants
from collections import deque
import xml.etree.ElementTree as ET

###########################
# Code to find connected components (relevant to extract IPS) in Aparc parcellation
###########################

def checkBounds(x, bounds):
    return x >= bounds[0] and x < bounds[1]

def appendNeighbours(xi, yi, zi, bounds, voxelMask, queue):
    triplet = [-1,0,1]
    for i in triplet:
        x = xi+i
        for j in triplet:
            y = yi+j
            for k in triplet:
                z = zi + k
                if checkBounds(x, bounds) and checkBounds(y, bounds) and checkBounds(z, bounds) and voxelMask[x,y,z]:
                    queue.append((x,y,z))
    return queue

def findConnectedComponentsFromMask(bounds, voxelMask):
    nodesX, nodesY, nodesZ = np.where(voxelMask)
    componentLabels = np.zeros(voxelMask.shape)

    label = 1
    for i in range(nodesX.size):
        xi, yi, zi = nodesX[i], nodesY[i], nodesZ[i]
        if voxelMask[xi, yi, zi]:
            # Create a queue of nearest neighbours and start while loop on neighbourhood to grow component
            queue = deque()
            queue.append((xi,yi,zi))
            while len(queue) > 0:
                # Pop a node from the queue
                x,y,z = queue.popleft()
                if voxelMask[x,y,z]:
                    componentLabels[x,y,z] = label
                    voxelMask[x,y,z] = False
                    queue = appendNeighbours(x,y,z, bounds, voxelMask, queue)
            label += 1
    return componentLabels

def extractIPSFromAparcAndMask(aparcSegPath, IPSmask):
    vol = nib.load(aparcSegPath)#"/media/miplab-nas2/Data/guibert/nBack_complete/working_dir/nBack_Share_HC/MRIDATA/1006/mri/aparc.a2009s+aseg.mgz")
    vol_data = vol.get_fdata()

    # First get the connected components in the IPS of the subject
    # In FreeSurfer's aparc segmentation, the IPS nodes are 12157 and 11157
    IPS_or = (vol_data == 12157) | (vol_data == 11157) | (vol_data == 12159) | (vol_data == 11159)
    print(IPS_or.sum())
    connectedCompIPS = findConnectedComponentsFromMask((0,vol.shape[0]), IPS_or)

    # Then get the atlas of the IPS from Glosser's parcellation.
    # Use it to select components having at least an intersection with this mask.
    ips_mask = IPSmask > 0 #"/media/miplab-nas2/Data/guibert/nBack_complete/working_dir/nBack_Share_HC/MRIDATA/1006/mri/IPS_atlas_coreg.nii.gz").get_fdata() > 0
    print(ips_mask.sum())
    print((connectedCompIPS > 0).sum())

    relevantComponents = np.unique(connectedCompIPS[ips_mask])
    print(relevantComponents)
    filtering_mask = np.zeros(connectedCompIPS.shape, dtype=bool)
    for s in relevantComponents:
        if s > 0:
            filtering_mask = filtering_mask | (connectedCompIPS == s)
    print(filtering_mask.sum())
    connectedCompIPS[np.logical_not(filtering_mask)] = 0
    
    # Once the IPS has been "intersected" we can map it back to its original hemispheric value.
    
    # All kept components are remapped to simply IPS left or right, using FreeSurfer's label convention
    compsR = np.unique(connectedCompIPS[vol_data == 12157])
    compsL = np.unique(connectedCompIPS[vol_data == 11157])

    componentsUniqueRight = np.setdiff1d(compsR, compsL)
    componentsUniqueLeft = np.setdiff1d(compsL, compsR)
    
    print(componentsUniqueRight)
    print(componentsUniqueLeft)

    for s in componentsUniqueRight:
        if s > 0:
            connectedCompIPS[connectedCompIPS == s] = 12157

    for s in componentsUniqueLeft:
        if s > 0:
            connectedCompIPS[connectedCompIPS == s] = 11157
    
    return nib.Nifti1Image(connectedCompIPS, vol.affine, vol.header)

#############################
# Code to extract from HCP atlas sets of regions and intersect with aparc
#############################

def getParcellationRegionsHCP(hcpPath):    
    # Get parcellation regions
    xmlParcellationHCP = ET.parse(hcpPath)#"/home/guibert/The-HCP-MMP1.0-atlas-in-FSL/HCP-Multi-Modal-Parcellation-1.0.xml")
    rootXml = xmlParcellationHCP.getroot()
    parcellationRegionsHCP = rootXml.findall('data/label')
    return parcellationRegionsHCP

def getRegionIDs(ROIname, regions):
    ids_ROI = []
    for region in regions:
        if ROIname in region.text:
            ids_ROI.append(int(region.get('index')))
    return ids_ROI

def getIPSregions(hcpData, aparcPath, regionsHCP):
    # Create IPS mask
    IPSmask = np.zeros(hcpData.shape, dtype=bool)
    
    # Get IPS ids
    ipsIDs = getRegionIDs('IPS', regionsHCP)
    for s in ipsIDs:
        IPSmask = IPSmask | (hcpData == int(s))
    # Refine IPS based on subject anatomy
    IPSrefinedAparc = extractIPSFromAparcAndMask(aparcPath, IPSmask)
    
    IPSl = IPSrefinedAparc.get_fdata() == 11157
    IPSr = IPSrefinedAparc.get_fdata() == 12157
    return IPSl, IPSr

def createPrecuneusMasks(aparcParcell, hcpData, parcellationRegionsHCP):
    # Get anterior precuneus as region
    # Precuneus limits in HCP atlas based on https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9895448/
    anteriorPCUL = getRegionIDs('L_7Am', parcellationRegionsHCP)
    anteriorPCUR = getRegionIDs('R_7Am', parcellationRegionsHCP)

    posteriorPCUL = getRegionIDs('L_POS2', parcellationRegionsHCP)
    posteriorPCUR = getRegionIDs('R_POS2', parcellationRegionsHCP)
    # Precuneus are 12130 and 11130
    # Get precuneus anterior fom the HCP mask, and split them as separate regions
    L_PCU_mask = aparcParcell == 11130
    R_PCU_mask = aparcParcell == 12130

    antPCUL_corr = L_PCU_mask & (hcpData == anteriorPCUL)
    antPCUR_corr = R_PCU_mask & (hcpData == anteriorPCUR)

    postPCUL_corr = L_PCU_mask & (hcpData == posteriorPCUL)
    postPCUR_corr = R_PCU_mask & (hcpData == posteriorPCUR)
    return antPCUL_corr, antPCUR_corr, postPCUL_corr, postPCUR_corr

def setAtlasWithMasks(atlasData, IPSL, IPSR, antPCUL, antPCUR, postPCUL, postPCUR):
    atlasData[antPCUL] = 39
    atlasData[antPCUR] = 39
    atlasData[postPCUL] = 40
    atlasData[postPCUR] = 40
    atlasData[IPSL] = 41
    atlasData[IPSR] = 42
    return atlasData

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

def correctAtlas(subjects_mri_dir, subject):
    mri_path = Path(subjects_mri_dir, subject, "mri")

    hcpAtlas = nib.load(str(Path(mri_path, 'hcp_atlas.nii.gz')))
    hcpData = hcpAtlas.get_fdata()
    
    parcellationRegionsHCP = getParcellationRegionsHCP("/home/guibert/The-HCP-MMP1.0-atlas-in-FSL/HCP-Multi-Modal-Parcellation-1.0.xml")
    # Extract IPS region from aparc parcellation
    IPSl, IPSr = getIPSregions(hcpData,
                           str(Path(mri_path,"aparc.a2009s+aseg.mgz")), 
                           parcellationRegionsHCP)
    
    # Extract precuneus
    aparcParcel = nib.load(str(Path(mri_path, "aparc.a2009s+aseg.mgz")))
    aparcData = aparcParcel.get_fdata()
    # Get precuneus regions, split between anterior and posterior and left and right
    antPCUL, antPCUR, postPCUL, postPCUR = createPrecuneusMasks(aparcData, hcpData, parcellationRegionsHCP)
    print("ant-PCUL:{}\nant-PCUR:{}\npost-PCUL:{}\npost-PCUR:{}\nIPSl:{}\nIPSr: {}".format(antPCUL.sum(),antPCUR.sum(),postPCUL.sum(),postPCUR.sum(), (IPSl>0).sum(), (IPSr>0).sum()))

    atlas = nib.load(str(get_atlas_coreg_fname(subjects_mri_dir, subject)))
    # Add to the atlas precuneus (anterior/posterior, left and right are merged) and intraparietal sulci (split in left/right)
    atlasData = setAtlasWithMasks(atlas.get_fdata(), IPSl, IPSr, antPCUL, antPCUR, postPCUL, postPCUR)

    # Save the result
    nib.Nifti1Image(atlasData, atlas.affine, atlas.header).to_filename(str(Path(mri_path, "atlas_coreg.nii.gz")))
    os.system("mri_convert {} {}".format(str(Path(mri_path, "atlas_coreg.nii.gz")),str(Path(mri_path, "atlas_coreg.mgz"))))
    
def put_atlas_to_subject_space(subject, subjects_mri_dir, t1_ref, atlas_path, hcp_atlas_path):
    mri_path = Path(subjects_mri_dir, subject, "mri")
    # Compute atlas space => subject space transform, using the T1_ref of the atlas
    mi = ants.image_read(str(t1_ref))
    fi = ants.image_read(str(Path(mri_path, "antsdn.brain.mgz")))
    mytx = ants.registration(fixed=fi, moving=mi, type_of_transform='SyN')
    warpedImage = ants.apply_transforms(fixed=fi, moving=mi, transformlist=mytx['fwdtransforms'])
    ants.image_write(warpedImage, str(Path(mri_path,  'T1atlas_coreg.nii.gz')))
    # Apply atlas => subject transform to atlas file
    print("Applying transform to atlas {}".format(atlas_path))
    mi = ants.image_read(str(atlas_path))
    warpedAtlas = ants.apply_transforms(fixed=fi, moving=mi, transformlist=mytx['fwdtransforms'],interpolator='nearestNeighbor')
    print("Saved transformed atlas to {}".format(str(get_atlas_coreg_fname(subjects_mri_dir, subject))))
    ants.image_write(warpedAtlas, str(get_atlas_coreg_fname(subjects_mri_dir, subject)))

    # Apply exactly the same transformation to the HCP atlas
    mi = ants.image_read(str(hcp_atlas_path))
    warpedHCP = ants.apply_transforms(fixed=fi, moving=mi, transformlist=mytx['fwdtransforms'],interpolator='nearestNeighbor')

    ants.image_write(warpedHCP, str(Path(mri_path,'hcp_atlas.nii.gz')))
    
def create_atlas_dict(subjects_mri_dir, subject_name, atlas_path):
    # Careful to extract and convert back as int. Otherwise labels will be floats, which results in errors!
    print("Using atlas: {}".format(atlas_path))
    atlas_vals = np.unique(nib.load(atlas_path).get_fdata())[1:].astype(np.uint8)
    region_dict = {}
    for i in range(38):
        region_dict["PCC_" + str(i)] = i+1
    region_dict["PCU_ant"] = 39
    region_dict["PCU_post"] = 40
    region_dict["IPS_l"] = 41
    region_dict["IPS_r"] = 42
    #for i in range(atlas_vals.size):
    #    region_dict["PCC_" + str(i)] = atlas_vals[i]
    # Atlas filename is expected to be in *.mgz, so convert back
    return (get_atlas_coreg_fname(subjects_mri_dir, subject_name), region_dict)