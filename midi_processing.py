import os
import sys
import pickle
import operator
import numpy as np
from collections import defaultdict
from collections import Counter 
from random import randint

import pretty_midi as pm
from tqdm import tqdm

# TODO: Stream class is no logner needed <- remore from code and make just SingleTrack.notes instead on SingleTrack.stream.notes
class Stream():
    
    def __init__ (self, first_tick, notes):
        self.notes = notes
        self.first_tick = first_tick
        
    def __repr__(self):
        return '<Stream object with {} musical events>'.format(len(self.notes))
      
class SingleTrack():
    '''class of single track in midi file encoded from pretty midi library
    
    atributes:
    ----------
        name:  str
            name of instrument class
        program: int
            midi instrument program
        is_drum: bool
            True if this track is drums track, False otherwise
        stream:
            Stream object of encoded music events (chords or notes)
    '''
    
    def __init__(self, name=None, program=None, is_drum=None, stream=None):
        self.name = name
        self.program = program
        self.is_drum = is_drum
        self.stream = stream
        self.is_melody = self.check_if_melody()
        
    def __repr__(self):
        return "<SingleTrack object. Name:{}, Program:{}, is_drum:{}>".format(self.name, self.program, self.is_drum)
    
    def to_pretty_midi_instrument(self, tempo=100):
        '''is create a pretty midi Instrument object from self.stream.notes sequance
           
            parameters: 
            -----------
                self: SingleTrack object
                
            return: 
            -------
                track: PrettyMIDI.Instrument object
        '''
        
        tempo_strech = 100/tempo
        track = pm.Instrument(program=self.program, is_drum=self.is_drum, name=self.name)
        time = self.stream.first_tick * tempo_strech
        for note in self.stream.notes:
            note_pitch = note[0]
            note_len = note[1] * tempo_strech
            for pitch in note_pitch:
                # if note is a rest (pause)
                if pitch == -1:
                    break
                event = pm.Note(velocity=100, pitch=pitch, start=time, end=time+note_len)
                track.notes.append(event)
            time = time + note_len

        return track
    
    def stream_to_bars(self, beat_per_bar=4):
        '''it takes notes and split it into equaly time distibuted sequances
        if note is between bars, the note is splited into two notes, with time sum equal to the note between bars.
        
        arguments:
        ----------
            stream: list of "notes"
            
        return:
        -------
            bars: list: list of lists of notes, every list has equal time. in musical context it returns bars
        '''
        # TODO: if last bar of sequance has less notes to has time equal given bar lenth it is left shorter
        # fill the rest of bar with rests
        
        # FIXME: there is a problem, where note is longer that bar and negative time occured
        # split note to max_rest_note, the problem occured when note is longer then 2 bars
        
        notes = self.stream.notes
        bars = []
        time = 0
        bar_index = 0
        add_tail = False
        note_pitch = lambda note: note[0]
        note_len = lambda note: note[1]
        for note in notes:
            try:
                temp = bars[bar_index]
            except IndexError:
                bars.append([])
                
            if add_tail:
                tail_pitch = note_pitch(tail_note)
                while tail_note_len > beat_per_bar:
                    bars[bar_index].append((tail_pitch, beat_per_bar))
                    tail_note_len -= beat_per_bar
                    bar_index += 1
                    bars.append([])
                       
                bars[bar_index].append((tail_pitch, tail_note_len))
                time += tail_note_len
                add_tail = False

            time += note_len(note)

            if time == beat_per_bar:
                bars[bar_index].append(note)
                time = 0
                bar_index += 1

            elif time > beat_per_bar: # if note is between bars
                between_bars_note_len =  note_len(note)
                tail_note_len = time - beat_per_bar
                leading_note_len = between_bars_note_len - tail_note_len
                
                leading_note = (note_pitch(note), leading_note_len)
                bars[bar_index].append(leading_note)
                tail_note = (note_pitch(note), tail_note_len)

                add_tail = True
                time = 0
                bar_index += 1
            else:
                bars[bar_index].append(note)
                
        return bars
    
    def check_if_melody(self):
        '''checks if Track object could be a melody
        
            it checks if percentage of single notes in Track.stream.notes is higher than treshold of 90%
            TODO: and there is at least 3 notes in bar per average
        
        '''
        events = None
        single_notes = None
        content_lenth = None
        
        for note in self.stream.notes:
            if self.name not in ['Bass','Drums']:
                events = 0
                content_lenth = 0
                single_notes = 0
                if note[0][0] != -1: # if note is not a rest
                    events += 1
                    content_lenth += note[1]
                    if len(note[0]) == 1: # if note is a single note, not a chord
                        single_notes += 1

        if events != None:
            if events == 0 or content_lenth == 0:
                return False
            else:
                single_notes_rate = single_notes/events
                density_rate = events/content_lenth
                if single_notes_rate >= 0.9 and density_rate < 2:
                    self.name = 'Melody'
                    return True
                else:
                    return False
        else:
            return False
          
          
class MultiTrack():
    '''Class that represent one midi file
    atributes:
        pm_obj : PrettyMIDI class object of this midi file
        res: resolution of midi
        time_to_tick: function that coverts miliseconds to ticks. it depends on midi resolution for every midi
        name: path to midi file
        tracks: a list of SingleTrack objects
    '''
    
    def __init__(self, path=None, tempo=100):
        self.tempo = tempo
        self.pm_obj = pm.PrettyMIDI(path, initial_tempo=self.tempo) # changename to self.PrettyMIDI
        self.res = self.pm_obj.resolution
        self.time_to_tick = self.pm_obj.time_to_tick
        self.name = path
        self.tracks = [parse_pretty_midi_instrument(instrument, self.res, self.time_to_tick, self.get_pitch_offset_to_C() ) for instrument in self.pm_obj.instruments]  
        self.tracks_by_instrument = self.get_track_by_instrument()
    
    # TODO: this function is deprecated <- remove from code
    def get_multiseq(self):
        '''tracks: list of SingleTrack objects
        reaturn a dictionary of sequences for every sequence in SingleTrack
        '''

        multiseq_indexes = set([key for music_track in self.tracks for key in music_track.seq])
        multiseq = dict()

        for seq_id in multiseq_indexes:
            multiseq[seq_id] = []

        for single_track in self.tracks:
            for key, value in single_track.seq.items():
                multiseq[key].append((single_track.name,value))

        return multiseq
      
    def get_programs(self, instrument):
        program_list = []
        for track in self.tracks:
          if track.name == instrument:
            program_list.append(track.program)
            
        return program_list
    
    def get_pitch_offset_to_C(self):
        '''to get better train resoult without augmenting midis to all posible keys
        we assumed that most frequent note is the rootnote of song then calculate
        the offset of semitones to move song key to C.
        
        You should ADD this offset to note pitch to get it right
        '''
        
        hist = self.pm_obj.get_pitch_class_histogram()
        offset = np.argmax(hist)
        if offset > 6:
            return 12-offset
        else:
            return -offset
    
    def save(self, path):
        midi_file = pm.PrettyMIDI()
        for track in self.tracks:
            midi_file.instruments.append(track.to_pretty_midi_instrument(self.tempo))
        midi_file.write(path)
        return midi_file
    
    def get_track_by_instrument(self):
        '''return a dictionary with tracks indexes grouped by instrument class'''
        tracks = self.tracks
        names = [track.name for track in tracks]
        uniqe_instruemnts = set(names)
        tracks_by_instrument = dict()
        for key in uniqe_instruemnts:
            tracks_by_instrument[key] = []

        for i, track in enumerate(tracks):
            tracks_by_instrument[track.name].append(i)

        return tracks_by_instrument
    
    def get_common_bars_for_every_possible_pair(self, x_instrument, y_instrument):
        ''' for every possible pair of given instrument classes
        returns common bars from multitrack'''
        x_bars = []
        y_bars = []
        pairs = self.get_posible_pairs(x_instrument, y_instrument)
        for x_track_index, y_track_index in pairs:
            _x_bars, _y_bars = get_common_bars(self.tracks[x_track_index], self.tracks[y_track_index])
            x_bars.extend(_x_bars)
            y_bars.extend(_y_bars)

        return x_bars, y_bars
    
    def get_data_seq2seq_arrangment(self, x_instrument, y_instrument, bars_in_seq=4):
        '''this method is returning a sequances of given lenth by rolling this lists of x and y for arrangemt generation
        x and y has the same bar lenth, and represent the same musical phrase playd my difrent instruments (tracks) 
        
        '''
        x_seq = []
        y_seq = []
        x_bars, y_bars = self.get_common_bars_for_every_possible_pair(x_instrument, y_instrument)

        for i in range(len(x_bars) - bars_in_seq + 1):
            x_seq_to_add = [note for bar in x_bars[i:i+bars_in_seq] for note in bar ]
            y_seq_to_add = [note for bar in y_bars[i:i+bars_in_seq] for note in bar ]
            x_seq.append(x_seq_to_add)
            y_seq.append(y_seq_to_add)

        return x_seq, y_seq
    
    def get_data_seq2seq_melody(self,instrument_class, x_seq_len=4):
        '''return a list of bars with content for every track with given instrument class for melody generaiton
        x_seq_len and y_seq_len
        
        x previous sentence, y next sentence of the same melody line
        
        '''

        instrument_tracks = self.tracks_by_instrument[instrument_class]

        for track_index in instrument_tracks:
            bars = self.tracks[track_index].stream_to_bars()
            bars_indexes_with_content = get_bar_indexes_with_content(bars)
            bars_with_content = [bars[i] for i in get_bar_indexes_with_content(bars)]

            x_seq = []
            y_seq = []
            for i in range(len(bars_with_content)-x_seq_len-1):
                _x_seq = [note for bar in bars_with_content[i:i+x_seq_len] for note in bar]
                _y_bar = bars_with_content[i+x_seq_len]
                x_seq.append(_x_seq)
                y_seq.append(_y_bar)

        return x_seq, y_seq
    
    def get_posible_pairs(self, instrument_x, instrument_y):
        '''it takes two lists, and return a list of tuples with every posible 2-element combination
        parameters:
        -----------
            instrument_x, instrument_y : string {'Guitar','Bass','Drums'} 
                a string that represent a instrument class you want to look for in midi file.

        returns:
        ----------
            pairs: list of tuples
                a list of posible 2-element combination of two lists
        '''
        x_indexes = self.tracks_by_instrument[instrument_x]
        y_indexes = self.tracks_by_instrument[instrument_y]
        pairs = [(x,y) for x in x_indexes for y in y_indexes]
        return pairs
    
    def show_map(self):
        print(self.name)
        print()
        for track in self.tracks:
            bars = track.stream_to_bars(4)
            track_str = ''
            for bar in bars:
                if bar_has_content(bar):
                    track_str += '█'
                else:
                    track_str += '_'

            print(track.name[:4],':', track_str)

            
def stream_to_bars(notes, beat_per_bar=4):
        '''it takes notes and split it into equaly time distibuted sequances
        if note is between bars, the note is splited into two notes, with time sum equal to the note between bars.
        arguments:
            stream: list of "notes"
        return:
            bars: list: list of lists of notes, every list has equal time. in musical context it returns bars
        '''
        # TODO: if last bar of sequance has less notes to has time equal given bar lenth it is left shorter
        # fill the rest of bar with rests
        
        # FIXME: there is a problem, where note is longer that bar and negative time occured
        # split note to max_rest_note, the problem occured when note is longer then 2 bars - FIXED
        
        bars = []
        time = 0
        bar_index = 0
        add_tail = False
        note_pitch = lambda note: note[0]
        note_len = lambda note: note[1]
        for note in notes:
            try:
                temp = bars[bar_index]
            except IndexError:
                bars.append([])
                
            if add_tail:
                tail_pitch = note_pitch(tail_note)
                while tail_note_len > beat_per_bar:
                    bars[bar_index].append((tail_pitch, beat_per_bar))
                    tail_note_len -= beat_per_bar
                    bar_index += 1
                       
                bars[bar_index].append((tail_pitch, tail_note_len))
                time += tail_note_len
                add_tail = False
            time += note_len(note)

            if time == beat_per_bar:
                bars[bar_index].append(note)
                time = 0
                bar_index += 1

            elif time > beat_per_bar: # if note is between bars
                between_bars_note_len =  note_len(note)
                tail_note_len = time - beat_per_bar
                leading_note_len = between_bars_note_len - tail_note_len
                leading_note = (note_pitch(note), leading_note_len)
                bars[bar_index].append(leading_note)
                tail_note = (note_pitch(note), tail_note_len)

                add_tail = True
                time = 0
                bar_index += 1
            else:
                bars[bar_index].append(note)
                
        return bars

def get_bar_len(bar):
    """calculate a lenth of a bar
    parameters:
        bar : list
            list of "notes", tuples like (pitches, len)
    """
    time = 0
    for note in bar:
        time += note[1]
    return time
  
def get_common_bars(track_x,track_y):
    '''return common bars, for two tracks is song
    return X_train, y_train list of 
    '''
    bars_x = track_x.stream_to_bars()
    bars_y = track_y.stream_to_bars()
    bwc_x = get_bar_indexes_with_content(bars_x)
    bwc_y = get_bar_indexes_with_content(bars_y)

    common_bars = bwc_x.intersection(bwc_y)
    common_bars_x = [bars_x[i] for i in common_bars]
    common_bars_y = [bars_y[i] for i in common_bars]
    return common_bars_x, common_bars_y
      
def get_bar_indexes_with_content(bars):
    '''this method is looking for non-empty bars in the tracks bars
    the empty bar consist of only rest notes.
    returns: a set of bars indexes with notes
    '''
    bars_indexes_with_content = set()
    for i, bar in enumerate(bars):
        if bar_has_content(bar):
            bars_indexes_with_content.add(i)

    return bars_indexes_with_content   

def get_bars_with_content(bars):
    '''this method is looking for non-empty bars in the tracks bars
    the empty bar consist of only rest notes.
    returns: a set of bars with notes
    '''
    bars_with_content = []
    for bar in bars:
        if bar_has_content(bar):
            bars_with_content.append(bar)

    return bars_with_content  
  
  
def bar_has_content(bar):
    '''check if bar has any musical information, more accurate
    it checks if in a bar is any non-rest event like note, or chord
    
    parameters:
    -----------
        bar: list
            list of notes
            
    return:
    -------
        bool:
            True if bas has concent and False of doesn't
    '''
    bar_notes = len(bar)
    count_rest = 0
    for note in bar:
        if note[0] == (-1,):
            count_rest += 1
    if count_rest == bar_notes:
        return False
    else:
        return True
      
def round_to_sixteenth_note(x, base=0.25):
        '''round value to closest multiplication by base
        in default to 0.25 witch is sisteenth note accuracy 
        '''
        
        return base * round(x/base)
       
def parse_pretty_midi_instrument(instrument, resolution, time_to_tick, key_offset):
    ''' arguments: a prettyMidi instrument object
        return: a custom SingleTrack object
    '''
       
    first_tick = None
    prev_tick = 0
    prev_note_lenth = 0
    max_rest_len = 4.0

    notes = defaultdict(lambda:[set(), set()])
    for note in instrument.notes:
        if first_tick == None:
            first_tick = 0
            
        tick = round_to_sixteenth_note(time_to_tick(note.start)/resolution)
        if prev_tick != None:
            act_tick = prev_tick + prev_note_lenth
            if act_tick < tick:
                rest_lenth = tick - act_tick
                while rest_lenth > max_rest_len:
                    notes[act_tick] = [{-1},{max_rest_len}]
                    act_tick += max_rest_len
                    rest_lenth -= max_rest_len
                notes[act_tick] = [{-1},{rest_lenth}]

        note_lenth = round_to_sixteenth_note(time_to_tick(note.end-note.start)/resolution)
        
        if -1 in notes[tick][0]:
            notes[tick] = [set(), set()]
        
        if instrument.is_drum:
            notes[tick][0].add(note.pitch)
        else:
            notes[tick][0].add(note.pitch + key_offset)
            
        notes[tick][1].add(note_lenth)

        prev_tick = tick
        prev_note_lenth = note_lenth
    
    notes = [(tuple(e[0]), max(e[1])) for e in notes.values()]

    name = 'Drums' if instrument.is_drum else pm.program_to_instrument_class(instrument.program)
    return SingleTrack(name, instrument.program, instrument.is_drum, Stream(first_tick,notes) )
  
def remove_duplicated_sequences(xy_tuple):
    ''' removes duplicated x,y sequences
    parameters:
    -----------
        xy_tuple: tuple of lists
            tuple of x,y lists that represens sequances in training set
            
    return:
    ------
        x_unique, y_unique: tuple
            a tuple of cleaned x, y traing set
    '''
    x = xy_tuple[0]
    y = xy_tuple[1]
    x_freeze = [tuple(seq) for seq in x]
    y_freeze = [tuple(seq) for seq in y]
    unique_data = list(set(zip(x_freeze,y_freeze)))
    x_unique = [seq[0] for seq in unique_data]
    y_unique = [seq[1] for seq in unique_data]
    return x_unique, y_unique
        
        
def extract_data(midi_folder_path=None, how=None, instrument=None, bar_in_seq=4, remove_duplicates=True):
    '''extract musical data from midis in given folder, to x_train, y_train lists on sequences
        
    parameters:
    -----------
        midi_folder_path : string 
            a path to directory where midi files are stored
        how : string {'melody','arrangment'}
            - if melody: function extract data of one instrument,
            and return lists of x and y that x is actual sequance of 4 bars
            and y is next bar
            - if arrangment: function extract data of two instruments and
            returns a lists of x and y that x is one instrument sequence,
            and y is coresponing sequance to x, played by second instrument
        instrument: string or tuple of two strings
            this parameter is used to specify a instrument class, or classes that you wanted
            to extract from midi files.
            
            if how='melody': string
            if how='arrangment' : (string_x, string_y)
            
    return:
    -------
        x_train, y_train - tuple of coresponding lists of x_train and y_train data for training set
        
    notes:
    ------
        extracted data is transposed to the key of C
        duplicated x,y pairs are removed
    '''
    if how not in {'melody','arrangment'}:
        raise ValueError('how parameter must by one of {melody, arrangment} ')

    x_train = []
    y_train = []
    
    programs_for_instrument = []
    
    from collections import Counter

    for directory, subdirectories, files in os.walk(midi_folder_path):
        for midi_file in tqdm(files, desc='Exporting: {}'.format(instrument)):
            midi_file_path = os.path.join(directory, midi_file)
            try:
                mt = MultiTrack(midi_file_path)
                # get programs
                mt.get_programs(instrument)
                
                if how=='melody':
                    x ,y = mt.get_data_seq2seq_melody(instrument, bar_in_seq)
                    programs_for_instrument.extend(mt.get_programs(instrument))
                if how=='arrangment':
                    x ,y = mt.get_data_seq2seq_arrangment(instrument[0], instrument[1], bar_in_seq)
                    programs_for_instrument.extend(mt.get_programs(instrument[1]))
                x_train.extend(x)
                y_train.extend(y)
            except:
                continue
                
    most_recent_program = most_recent(programs_for_instrument)
    
    if remove_duplicates:   
        x_train, y_train = remove_duplicated_sequences((x_train, y_train))
        
    return x_train , y_train, most_recent_program
  
def most_recent(list): 
    occurence_count = Counter(list) 
    return occurence_count.most_common(1)[0][0] 
  
def analyze_data(midi_folder_path):
    '''Show usage of instumets in midipack
    
    parameters:
    -----------
        midi_folder_path : string 
            a path to directory where midi files are stored
    '''
    
    instrument_count = dict()
    instrument_programs = dict()
    
    for directory, subdirectories, files in os.walk(midi_folder_path):
        for midi_file in tqdm(files):
            midi_file_path = os.path.join(directory, midi_file)
            try:
                mt = MultiTrack(midi_file_path)
                for track in mt.tracks:
                    try:
                        instrument_count[track.name] += len(get_bars_with_content(track.stream_to_bars()))
                    except KeyError:
                        instrument_count[track.name] = 1
            except Exception as e:
                print(e)
    
    for key, value in sorted(instrument_count.items(), key=lambda x: x[1], reverse=True):
        print(value, 'of', key)
        