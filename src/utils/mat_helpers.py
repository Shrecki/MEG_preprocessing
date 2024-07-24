from mne.io import RawArray
import pymatreader
import pandas as pd

def sanitize_empty(a):
    if a.size > 0:
        a = a[0]
    else:
        a = 0
    return a

def correct_data_units(data, units):
    for i in range(len(units)):
        data[i,:] *= convert_unit(units[i])
    return data
    

def create_raw_from_mat_and_tsv(header_mat, payload_csv):
    info, units, ch_types = extract_info_from_mat(header_mat)
    df = pd.read_csv(payload_csv, sep='\t', header=None, dtype=np.double)
    data = df.to_numpy().T
    
    # The data to be passed should be in proper units.
    array = RawArray(correct_data_units(data, units), info, verbose=1)
    return array