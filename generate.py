from midi_processing import MultiTrack, SingleTrack, Stream
from model import Seq2SeqModel, seq_to_numpy
from tqdm import tqdm
import argparse
import os
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('n', help='name for experiment', type=str)
parser.add_argument('s', help='session name', type=str)
parser.add_argument('--i', help='number of midis to generate', type=int)
# parser.add_argument('--l', help='latent_dim_of_model', type=int)
parser.add_argument('--m', help="mode {'from_seq', 'from_state}'", type=str)
args = parser.parse_args()

EXPERIMENT_NAME = args.n
SESSION_NAME = args.s
GENERETIONS_COUNT = args.i
# LATENT_DIM = args.l
MODE = args.m

if not GENERETIONS_COUNT:
    GENERETIONS_COUNT = 1

# if not LATENT_DIM:
#     LATENT_DIM = 256
    
if not MODE:
    MODE = 'from_seq'

model_workflow = pickle.load(open(os.path.join('training_sets', EXPERIMENT_NAME, 'workflow.pkl'),'rb'))

band = dict()
for key, value in model_workflow.items():
    if isinstance(value[0], str):
        instrument = value[0]
        generator = None
    else:
        instrument = value[0][1]
        generator = value[0][0]
            
    band[instrument] = [None, None, generator]

'''LOAD MODELS'''
print('Loading models...')
for instrument in tqdm(band):
    
    data_path = os.path.join('training_sets', EXPERIMENT_NAME, instrument.lower() + '_data.pkl')
    model_path = os.path.join('models', EXPERIMENT_NAME, instrument.lower() + '_model.h5')
    
    x_train, y_train, program, bars_in_seq = pickle.load(open(data_path,'rb'))
    model = Seq2SeqModel(x_train, y_train, bars_in_seq=bars_in_seq)
    model.load(model_path)
    band[instrument][0] = model
    band[instrument][1] = program

print('Generating music...')
for midi_counter in tqdm(range(GENERETIONS_COUNT)):
    ''' MAKE MULTIINSTRUMENTAL MUSIC !!!'''
    notes = dict()

    for instrument, (model, program, generator) in band.items():
        if generator == None:
            notes[instrument] = model.develop(mode=MODE)
        else:
            input_data = seq_to_numpy(notes[generator],
                            model.transformer.x_max_seq_length,
                            model.transformer.x_vocab_size,
                            model.transformer.x_transform_dict)
            notes[instrument] = model.predict(input_data)[:-1]

    '''COMPILE TO MIDI'''
    generated_midi = MultiTrack()
    for instrument, (model, program, generator) in band.items():
        if instrument == 'Drums':
            is_drums = True
        else:
            is_drums = False

        stream = Stream(first_tick=0, notes=notes[instrument])
        track = SingleTrack(name=instrument ,program=program, is_drum=is_drums, stream=stream)
        generated_midi.tracks.append(track)

    # make folder for new experiment
    try:
        os.mkdir(os.path.join('generated_music', EXPERIMENT_NAME))
    except:
        pass
    try:
        os.mkdir(os.path.join('generated_music', EXPERIMENT_NAME, SESSION_NAME))
    except:
        pass

    save_path = os.path.join('generated_music', EXPERIMENT_NAME, SESSION_NAME, f'{EXPERIMENT_NAME}_{midi_counter}_{MODE}.mid')
    generated_midi.save(save_path)
    # print(f'Generated succefuly to {save_path}')
