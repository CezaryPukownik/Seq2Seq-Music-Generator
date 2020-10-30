import os
import sys
import pickle
import keras
import argparse
import warnings
import pandas as pd
from model import Seq2SeqModel
from extract import make_folder_if_not_exist

# TODO: 
# FIXME:

def parse_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument('n', help='name for experiment', type=str)
    parser.add_argument('--b', help='batch_size', type=int)
    parser.add_argument('--l', help='latent_dim', type=int)
    parser.add_argument('--e', help='epochs', type=int)
    parser.add_argument('--ed', help='encoder dropout', type=float)
    parser.add_argument('--dd', help='decoder dropout', type=float)
    parser.add_argument('--i', help='refrance to instrument to train, if you want to train only one instument')
    parser.add_argument('-r', help='reset, use when you want to reset waights and train from scratch', action='store_true')
    args = parser.parse_args()
    return args

def load_workflow():
    workflow_path = os.path.join('training_sets', EXPERIMENT_NAME, 'workflow.pkl')
    if os.path.isfile(workflow_path):
        model_workflow = pickle.load(open(workflow_path,'rb'))
    else:
        raise FileNotFoundError(f'There is no workflow.pkl file in trainig_sets/{EXPERIMENT_NAME}/ folder')
    return model_workflow

def train_models(model_workflow):
    
    instruments = [instrument if how == 'melody' else instrument[1] for key, (instrument, how) in model_workflow.items()]
    # make_folder_if_not_exist(os.mkdir(os.path.join('models',EXPERIMENT_NAME)))
    
    found = False
    for instrument in instruments:

        if not INSTRUMENT or INSTRUMENT == instrument:
            data_path = os.path.join('training_sets', EXPERIMENT_NAME, instrument.lower() + '_data.pkl')
            model_path = os.path.join('models', EXPERIMENT_NAME, f'{instrument.lower()}_model.h5')
            history_path = os.path.join('models', EXPERIMENT_NAME, f'{instrument.lower()}_history.csv')

            x_train, y_train, _, bars_in_seq = pickle.load(open(data_path,'rb'))
            
            if os.path.isfile(model_path) and not RESET:
                model = Seq2SeqModel(x_train, y_train)
                model.load(model_path)
            else:
                model = Seq2SeqModel(x_train, y_train, LATENT_DIM, ENCODER_DROPOUT, DECODER_DROPOUT, bars_in_seq)

            print(f'Training: {instrument}')
            history = model.fit(BATCH_SIZE, EPOCHS, callbacks=[])
            make_folder_if_not_exist(os.path.join('models', EXPERIMENT_NAME))
            pd.DataFrame(history.history).to_csv(history_path, mode='a', header=False)
            model.save(model_path)
            found = True

    if not found:
        raise ValueError(f'Instrument not found. Use one of the {instruments}')
    
if __name__ == '__main__':

    warnings.filterwarnings("ignore")
    args = parse_argv()
    
    EXPERIMENT_NAME = args.n
    BATCH_SIZE = args.b
    LATENT_DIM = args.l
    EPOCHS = args.e
    RESET = args.r
    INSTRUMENT = args.i
    ENCODER_DROPOUT = args.ed
    DECODER_DROPOUT = args.dd

    # default settings if not args passed
    if not BATCH_SIZE:
        BATCH_SIZE = 32
    if not LATENT_DIM:
        LATENT_DIM = 256
    if not EPOCHS:
        EPOCHS = 1
    if not RESET:
        RESET = False
    if not ENCODER_DROPOUT:
        ENCODER_DROPOUT = 0.0
    if not DECODER_DROPOUT:
        DECODER_DROPOUT = 0.0
    
    train_models(load_workflow())
