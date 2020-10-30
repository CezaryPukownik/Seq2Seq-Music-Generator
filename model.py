from __future__ import print_function
from midi_processing import stream_to_bars
from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense, LSTM, LSTMCell, TimeDistributed
import numpy as np

class Seq2SeqTransformer():
    ''' encoder/transforer 
    params:
    -------
        x_train, y_train - list of sequences

    methods:
        fit
        transform'''
    
    def __init__(self):
        self.transform_dict = None
        self.reverse_dict = None
        self.vocab_x = None
        self.vocab_y = None
        
        
    def preprocess(self, x_train, y_train):
        '''Converts training set do list and add special chars'''

        _x_train = []
        for i, seq in enumerate(x_train):
            _x_train.append([])
            for note in seq:
                _x_train[i].append(note)

        _y_train = []
        for i, seq in enumerate(y_train):
            _y_train.append([])
            _y_train[i].append('<GO>')
            for note in seq:
                    _y_train[i].append(note)
            _y_train[i].append('<EOS>')
                    
        return _x_train, _y_train

    def transform(self, x_train, y_train):
        
        x_vocab = set([note for seq in x_train for note in seq])
        y_vocab = set([note for seq in y_train for note in seq])
        
        self.x_vocab = sorted(list(x_vocab)) 
        self.y_vocab = ['<GO>','<EOS>']
        self.y_vocab.extend(sorted(list(y_vocab)))

        self.x_vocab_size = len(self.x_vocab)
        self.y_vocab_size = len(self.y_vocab)

        self.x_transform_dict = dict(
            [(char, i) for i, char in enumerate(self.x_vocab)])
        self.y_transform_dict = dict(
            [(char, i) for i, char in enumerate(self.y_vocab)])
        self.x_reverse_dict = dict(
            (i, char) for char, i in self.x_transform_dict.items())
        self.y_reverse_dict = dict(
            (i, char) for char, i in self.y_transform_dict.items())
               
        x_train, y_train = self.preprocess(x_train, y_train)
        
        self.x_max_seq_length = max([len(seq) for seq in x_train])
        self.y_max_seq_length = max([len(seq) for seq in y_train])
        
        encoder_input_data = np.zeros(
            (len(x_train), self.x_max_seq_length, self.x_vocab_size),
            dtype='float32')
        decoder_input_data = np.zeros(
            (len(x_train), self.y_max_seq_length, self.y_vocab_size),
            dtype='float32')
        decoder_target_data = np.zeros(
            (len(x_train), self.y_max_seq_length, self.y_vocab_size),
            dtype='float32')

        for i, (x_train, y_train) in enumerate(zip(x_train, y_train)):
            for t, char in enumerate(x_train):
                encoder_input_data[i, t,  self.x_transform_dict[char]] = 1.
            for t, char in enumerate(y_train):
                decoder_input_data[i, t,  self.y_transform_dict[char]] = 1.
                if t > 0:
                    decoder_target_data[i, t - 1,  self.y_transform_dict[char]] = 1.
                    
        return encoder_input_data, decoder_input_data, decoder_target_data
    
    
      
      
class Seq2SeqModel():
    '''NeuralNerwork Seq2Seq model.
    The network is created based on training data
    '''

    def __init__(self, x_train, y_train, latent_dim=256, enc_dropout=0, dec_dropout=0, bars_in_seq=4):
        self.has_predict_model = False
        self.has_train_model = False
        self.x_train = x_train
        self.y_train = y_train
        self.bars_in_seq = bars_in_seq
        self.latent_dim = latent_dim
        self.transformer = Seq2SeqTransformer()
        self.encoder_input_data, self.decoder_input_data, self.decoder_target_data = self.transformer.transform(self.x_train, self.y_train)
    
        # ---------------
        # SEQ 2 SEQ MODEL:
        #     INPUT_1 : encoder_input_data
        #     INPUT_2 : decodet_input_data
        #     OUTPUT : decoder_target_data
        # ---------------
        
        # ENCODER MODEL
        #---------------
        
        # 1 layer - Input : encoder_input_data
        self.encoder_inputs = Input(shape=(None, self.transformer.x_vocab_size ))
        
        # 2 layer - LSTM_1, LSTM
        self.encoder = LSTM(latent_dim, return_state=True, dropout=enc_dropout)
        #self.encoder = LSTM(latent_dim, return_state=True)
        
        # 2 layer - LSTM_1 : outputs
        self.encoder_outputs, self.state_h, self.state_c = self.encoder(self.encoder_inputs)
        self.encoder_states = [self.state_h, self.state_c]  
        
       
        # DECODER MODEL
        #---------------
        
        # 1 layer - Input : decoder_input_data
        self.decoder_inputs = Input(shape=(None, self.transformer.y_vocab_size)) 
        
        # 2 layer - LSTM_1, LSTM
        self.decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=dec_dropout)
        #self.decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
        
        # 2 layer - LSTM_2 : outputs, full sequance as lstm layer
        self.decoder_outputs, _, _ = self.decoder_lstm(self.decoder_inputs,                     
                                                  initial_state=self.encoder_states)
        
        # 3 layer - Dense
        self.decoder_dense = Dense(self.transformer.y_vocab_size, activation='softmax')
        
        # 3 layer - Dense : outputs, full sequance as the array of one-hot-encoded elements 
        self.decoder_outputs = self.decoder_dense(self.decoder_outputs)
        
    def init_train_model(self):
        self.train_model = Model([self.encoder_inputs, self.decoder_inputs], self.decoder_outputs)
        self.train_model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        
    def fit(self, batch_size, epochs, callbacks):     
        
        if not self.has_train_model:
            self.init_train_model()
            self.has_train_model = True
        
        
        history = self.train_model.fit([self.encoder_input_data, self.decoder_input_data], self.decoder_target_data,
                              batch_size=batch_size,
                              epochs=epochs,
                              callbacks=callbacks,
                              validation_split=0.2)
        return history
        
    def save(self, path):
        self.train_model.save(path)
    
    def load(self, path):
        self.train_model = load_model(path)
        self.has_train_model = True
        
        self.encoder_inputs = self.train_model.layers[0].input
        self.encoder = self.train_model.layers[2]
        self.encoder_outputs, self.state_h, self.state_c = self.train_model.layers[2].output
        self.encoder_states = [self.state_h, self.state_c]  
        self.decoder_inputs = self.train_model.layers[1].input
        self.decoder_lstm = self.train_model.layers[3]
        self.decoder_outputs, _, _ = self.train_model.layers[3].output
        self.decoder_dense = self.train_model.layers[4]
        self.decoder_outputs = self.train_model.layers[4].output
            
    def init_predict_model(self):
        
        # ENCODER MODEL <- note used in develop music process
        #     from encoder_input to encoder_states
        #     to give a context to decoder model
        #---------------------------------
        
        self.encoder_model = Model(self.encoder_inputs, self.encoder_states)

        
        # DECODER MODEL
        #     From states (context) to sequance by generating firts element from context vector
        #     and starting element <GO>. Then adding predicted element as input to next cell, with
        #     updated states (context) by prevously generated element.
        #   
        #     INPUT_1 : state_h
        #     INPUT_2 : state_c
        #     INPUT_3 : y_train sized layer, that will be recursivly generated starting from <GO> sign
        #     
        #     INPUT -> LSTM -> DENSE
        #
        #     OUTPUT : one-hot-encoded element of sequance
        #     OUTPUT : state_h (updated)
        #     OUTPUT : state_c (updated)
        # -------------
        
        # 1 layer: TWO INPUTS: decoder_state_h, decoder_state_c
        self.decoder_state_input_h = Input(shape=(self.latent_dim,))  
        self.decoder_state_input_c = Input(shape=(self.latent_dim,))
        self.decoder_states_inputs = [self.decoder_state_input_h, self.decoder_state_input_c]
        

        # 2 layer: LSTM_1 output: element of sequance, lstm cell states
        self.decoder_outputs, self.state_h, self.state_c = self.decoder_lstm(
            self.decoder_inputs,
            initial_state = self.decoder_states_inputs
            )
        
        self.decoder_states = [self.state_h, self.state_c]
        
        # 3 layer: Dense output: one-hot-encoded representation of element of sequance
        self.decoder_outputs = self.decoder_dense(self.decoder_outputs)

        self.decoder_model = Model(                               
            [self.decoder_inputs] + self.decoder_states_inputs,
            [self.decoder_outputs] + self.decoder_states)
        self.has_predict_model = True
        
    def predict(self, input_seq=None, mode=None):
        
        if not self.has_predict_model:
            self.init_predict_model()
            self.has_predict_model = True
        
        if mode == 'generate':
            # create a random context as starting point
            h = np.random.rand(1,self.latent_dim)*2 - 1
            c = np.random.rand(1,self.latent_dim)*2 - 1
            states_value = [h, c]
        else:
            # get context from input sequance
            states_value = self.encoder_model.predict(input_seq)

        # make the empty decoder_input_data
        # and create the starting <GO> element of decoder_input_data
        target_seq = np.zeros((1, 1, self.transformer.y_vocab_size))  
        target_seq[0, 0, self.transformer.y_transform_dict['<GO>']] = 1.

        # sequance generation loop of decoder model
        stop_condition = False
        decoded_sentence = []
#         time = 0
        while not stop_condition:            
            
            # INPUT_1 : target_seq : started from empty array with start <GO> char
            # and recursivly updated by predicted elements
            
            # INPUT_2 : states_value :context from encoder model or randomly generated in develop mode
            # this can give as a 2 * latent_dim parameters to play with in manual generation
            
            # OUTPUT_1 : output_tokens : one hot encoded predicted element of sequance
            # OUTPUT_2,3 : h, c : context updated by predicted element
            output_tokens, h, c = self.decoder_model.predict(
                [target_seq] + states_value)

            # get most likly element index
            # translate from index to final (in normal form) preidcted element
            # append it to output list
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.transformer.y_reverse_dict[sampled_token_index]
            decoded_sentence.append(sampled_char)
            
#             time += sampled_char[1]
            # or time>=16
            if (sampled_char == '<EOS>' or len(decoded_sentence) > self.transformer.y_max_seq_length ):
                stop_condition = True

            target_seq = np.zeros((1, 1, self.transformer.y_vocab_size))
            target_seq[0, 0, sampled_token_index] = 1.

            states_value = [h, c]

        return decoded_sentence
    
    def develop(self, mode='from_seq'):
        
        # music generation for seq2seq for melody
        # TODO: Hardcoded 16 ??
        input_seq_start = random_seed_generator(self.bars_in_seq * 4, 
                                                self.transformer.x_max_seq_length,
                                                self.transformer.x_vocab_size,
                                                self.transformer.x_transform_dict,
                                                self.transformer.x_reverse_dict)
        
        input_data = seq_to_numpy(input_seq_start,
                                  self.transformer.x_max_seq_length,
                                  self.transformer.x_vocab_size,
                                  self.transformer.x_transform_dict)

        # generate sequnce iterativly for melody
        input_seq = input_seq_start.copy()
        melody = []
        for i in range(self.bars_in_seq):
            if mode == 'from_seq':
                decoded_sentence = self.predict(input_data)[:-1]
            elif mode == 'from_state':
                decoded_sentence = self.predict(mode='generate')[:-1]
            else:
                raise ValueError('mode must be in {from_seq, from_state}')
            melody.append(decoded_sentence)
            input_seq.extend(decoded_sentence)
            input_bars = stream_to_bars(input_seq, self.bars_in_seq)
            input_bars = input_bars[1:self.bars_in_seq+1]
            input_seq = [note for bar in input_bars for note in bar]
            input_data = seq_to_numpy(input_seq,
                                      self.transformer.x_max_seq_length,
                                      self.transformer.x_vocab_size,
                                      self.transformer.x_transform_dict)

        melody = [note for bar in melody for note in bar]
        return melody
      
def random_seed_generator(time_of_seq, max_encoder_seq_length, num_encoder_tokens, input_token_index, reverse_input_char_index):
    time = 0
    random_seq = []
    items = 0
    stop_sign = False
    while (time < time_of_seq):
        seed = np.random.randint(0,num_encoder_tokens-1)
        note = reverse_input_char_index[seed]
        time += note[1]
        if time > time_of_seq:
            note_time = note[1] - (time-time_of_seq)
            trimmed_note = (note[0],note_time)
            try:
                seed = input_token_index[trimmed_note]
                random_seq.append(trimmed_note)
                items += 1
            except KeyError:
                time -= note[1]
                continue
        else:
            random_seq.append(note)
            items += 1
        
        if items > max_encoder_seq_length:
            time = 0
            random_seq = []
            items = 0
            stop_sign = False
        
    return random_seq  
  
# seq to numpy array:
def seq_to_numpy(seq, max_encoder_seq_length, num_encoder_tokens, input_token_index):
    input_data = np.zeros(
        (1, max_encoder_seq_length, num_encoder_tokens),
        dtype='float32')

    for t, char in enumerate(seq):
        try:
            input_data[0, t, input_token_index[char]] = 1.
        except KeyError:
            char_time = char[1]
            _char = ((-1,), char_time)
        except IndexError:
            break

    return input_data