import os
import sys
import argparse
import pickle

from midi_processing import extract_data, analyze_data

def make_folder_if_not_exist(path):
    try:
        os.mkdir(path)
    except:
        pass

def parse_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument('midi_pack', help='folder name for midi pack in midi_packs folder', type=str)
    parser.add_argument('--n', help='name for experiment', type=str)
    parser.add_argument('--b', help='lengh of sequence in bars', type=int)
    parser.add_argument('-a', help='analize data', action='store_true')
    args = parser.parse_args()
    return args

def ask_for_workflow():
    '''MODEL WORKFLOW DIALOG'''
    number_of_instruments = int(input('Please specify number of instruments\n'))
    model_workflow = dict()
    for i in range(number_of_instruments):
        input_string = input('Please specify a workflow step <Instrument> [<Second Instrument>] <mode> {m : melody, a : arrangment}\n')
        tokens = input_string.split()
        if tokens[-1] == 'm':
            model_workflow[i] = (tokens[0], 'melody')
        elif tokens[-1] == 'a':
            model_workflow[i] = ((tokens[1], tokens[0]), 'arrangment')
        else:
            raise ValueError("The step definitiom must end with {'m', 'a'}");
    
    make_folder_if_not_exist(os.path.join('training_sets', EXPERIMENT_NAME))
    pickle.dump(model_workflow, open(os.path.join('training_sets', EXPERIMENT_NAME, 'workflow.pkl'),'wb'))
    
    return model_workflow

def extract_from_folder(model_workflow):
    for key, (instrument, how) in model_workflow.items():
        if how == 'melody':
            instrument_name = instrument
        else:
            instrument_name = instrument[1]
    
        make_folder_if_not_exist(os.path.join('training_sets', EXPERIMENT_NAME))
        save_path = os.path.join('training_sets', EXPERIMENT_NAME, instrument_name.lower() + '_data.pkl')
        
        x_train, y_train, program = extract_data(midi_folder_path=os.path.join('midi_packs', MIDI_PACK_NAME),
                                                 how=how, 
                                                 instrument=instrument,
                                                 bar_in_seq=BARS_IN_SEQ)
        
        pickle.dump((x_train, y_train, program, BARS_IN_SEQ), open(save_path,'wb'))

if __name__ == '__main__':
    args = parse_argv()

    MIDI_PACK_NAME = args.midi_pack
    EXPERIMENT_NAME = args.n
    BARS_IN_SEQ = args.b
    if not EXPERIMENT_NAME:
        EXPERIMENT_NAME = MIDI_PACK_NAME
    if not BARS_IN_SEQ:
        BARS_IN_SEQ = 4
    ANALIZE = args.a

    if ANALIZE:
        analyze_data(os.path.join('midi_packs', MIDI_PACK_NAME))
    else:
        extract_from_folder(ask_for_workflow())