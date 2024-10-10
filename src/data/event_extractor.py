import numpy as np
import mne
import os

annots_dict={"0-back T_W": 1, "0-back T_C": 2, "1-back T_W": 3, "1-back T_C": 4,
             "2-back T_W": 5, "2-back T_C": 6, "0-back D_W": 7, "0-back D_C": 8,
             "1-back D_W": 9, "1-back D_C": 10, "2-back D_W": 11, "2-back D_C": 12}

rev_annots = { annots_dict[k]:k for k in annots_dict.keys()}



def confusion_mat_by_condition(event_types, event_counts, event_dict):
    conditions = ['0-back', '1-back', '2-back']
    rates = np.zeros((len(conditions),4))
    
    for i,c in enumerate(conditions):
        rates[i,0] = sanitize_empty(event_counts[event_types == event_dict[c + ' ' + 'T_C']])
        rates[i,1] = sanitize_empty(event_counts[event_types == event_dict[c + ' ' + 'T_W']])
        rates[i,2] = sanitize_empty(event_counts[event_types == event_dict[c + ' ' + 'D_C']])
        rates[i,3] = sanitize_empty(event_counts[event_types == event_dict[c + ' ' + 'D_W']])
    return rates


"""
To convert to stimulus, we must first establish translation list of last 3 bits of the stimulus value:

101 => 0 back target
100/001 (as 100 doesn't exist?) => 0 back distractor
011 => 1 back target
010 => 1 back distractor
111 => 2 back target
110 => 2 back distractor
"""
def get_mapping_from_ts(ts):
    unique_vals = np.unique(ts)
    stim_corresp = {}
    stim_types = ["101", "001", "110", "010", "111", "011"]
    for v in unique_vals:
        bin_rep = np.binary_repr(v)[-3:]
        stim_rep = -1
        if bin_rep in stim_types:
            stim_rep = stim_types.index(bin_rep)
            stim_corresp[v] = stim_rep
    return stim_corresp

def convert_from_single_stim_to_multi_stims(raw_file_stim_ts):
    ts = np.int64(raw_file_stim_ts) #raw_file['STI101'][0])[0]
    stim_corresp = get_mapping_from_ts(ts)
    # Now we must convert the time serie to its id. Easy enough, right?
    ts_new = np.zeros((6, ts.size))
    for k in stim_corresp.keys():
        ts_new[stim_corresp[k], ts == k] = 1
    return ts_new, stim_corresp

def return_valid_events_from_probe(raw_file):
    converted_events, stim_corresp = convert_from_single_stim_to_multi_stims(raw_file['STI101'][0][0])
    # Get original events
    events = mne.find_events(raw_file, "STI101")
    # Translate each event to its unique value!
    for i in range(events.shape[0]):
        ev = events[i,2]
        r = np.binary_repr(ev)
        if len(r) >=8 and r[-8] == "1" and len(r) < 12 and ev in stim_corresp.keys():
            events[i, 2] = stim_corresp[ev] + 1
        else:
            events[i, 2] = 0
    filt_events = events[events[:,2] != 0 ]
    print(filt_events[:,2].size)
    filt_events = filt_events[np.diff(filt_events[:,0], append=2800) >= 1000]
    print(filt_events[:,2].size)
    return filt_events

def get_events_from_raw(raw_path, raw, stimulus_channel):
    stim_signal = raw[stimulus_channel][0]
    stim_fname = str(raw_path).replace('.fif', '_stims.csv')
    #print(stim_signal)
    #print(stim_fname)
    np.savetxt(stim_fname,stim_signal,delimiter=',')
    
    # Create the event table with a MATLAB script written by Costners
    events_fname = str(raw_path).replace('.fif', '_events.tsv')
    script_name = "/media/RCPNAS/Data/guibert/nBack_complete/nBack_Share_HC/nBackTriggers_modif_Fab.m"
    script_path = "/media/RCPNAS/Data/guibert/nBack_complete/nBack_Share_HC"
    cmd="matlab -nodisplay -nosplash -nodesktop -r \"cd {}; nBackTriggers_modif_Fab('{}', '{}', 'N'); exit\"".format(script_path, stim_fname, events_fname)
    os.system(cmd)
    # Get back the table
    event_table = np.loadtxt(events_fname, delimiter=',')
    
    # Delete the temporary files
    #os.remove(stim_fname)
    #os.remove(events_fname)
    return event_table

def load_event_table_corrected(events_fname):
    event_table = np.loadtxt(events_fname, delimiter=',')
    event_dictionary={1:"0-back T", 2: "1-back T", 3: "2-back T", 4: "0-back D", 5: "1-back D", 6: "2-back D"}
    events_annotated = [""]*event_table.shape[0]
    for i in range(len(events_annotated)):
        event_id = int(event_table[i,0])
        suffix = "_C" if (event_id < 4 and event_table[i,4] > 0) or (event_id >= 4 and event_table[i,4] == 0) else "_W"
        events_annotated[i] = event_dictionary[event_id] + suffix
    return events_annotated, event_table


def convert_event_table_to_annots(event_table, orig_time, first_samp):
    event_dictionary={1:"0-back T", 2: "1-back T", 3: "2-back T", 4: "0-back D", 5: "1-back D", 6: "2-back D"}
    events_annotated = [""]*event_table.shape[0]
    for i in range(len(events_annotated)):
        event_id = int(event_table[i,0])
        suffix = "_C" if (event_id < 4 and event_table[i,4] > 0) or (event_id >= 4 and event_table[i,4] == 0) else "_W"
        events_annotated[i] = event_dictionary[event_id] + suffix
    annotations = mne.Annotations(onset=event_table[:, 2] + first_samp, duration=event_table[:, 3], description=events_annotated, orig_time=orig_time)
    # Make sure all annotations are valid and not some spurious artifacts!
    annotations = annotations[annotations.duration > 1]
    return annotations


def get_annotations_from_raw(raw_path, raw, stimulus_channel, orig_time):
    return convert_event_table_to_annots(get_events_from_raw(raw_path, raw, stimulus_channel), orig_time, raw.first_time)

def annotate_raw_with_events_from_table(raw_path, raw):
    # Ensures already existing annotations are not thrown away
    hand_annots = raw.annotations
    orig_time = hand_annots.orig_time if hand_annots is not None else None
    annots_events = get_annotations_from_raw(raw_path, raw, "STI101", orig_time)
    if hand_annots is not None:
        annots_events.append(hand_annots.onset, hand_annots.duration, hand_annots.description)
        #hand_annots.append(annots_events.onset, annots_events.duration, annots_events.description)
    raw.set_annotations(annots_events)
    return raw

def annotate_and_get_events_from_annots(raw, annots):
    # get potential annotations first, to ensure we don't overwrite bad channel annotations for example
    raw_annots = raw.annots
    raw.set_annotations(annots)
    return get_events_from_annotated_raw(raw)

def get_events_from_annotated_raw(raw):
    events, event_ids = mne.events_from_annotations(raw, event_id=annots_dict)
    return events, event_ids