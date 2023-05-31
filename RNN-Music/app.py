from pydub.playback import play
import numpy as np 
import fluidsynth
import tensorflow as tf
from tensorflow import keras
from pydub import AudioSegment
from pydub.playback import play 
import pretty_midi
import mido
import os 


def convert_notes_to_numbers(notes):
    unique_notes = set(notes)
    note_to_number = {note: number for number, note in enumerate(unique_notes)}
    encoded_notes = [note_to_number[note] for note in notes]
    return encoded_notes

def training_data(encoded_notes, sequence_length):

    sequences = []
    next_notes = []
    for i in range(0, len(encoded_notes) - sequence_length, step):
        sequences.append(encoded_notes[i:i + sequence_length])
        next_notes.append(encoded_notes[i + sequence_length])
    
    X_train = np.array(sequences)
    y_train = np.array(next_notes)

    X_train = X_train / float(num_features)

    return X_train, y_train 




midi_directory = "./dataset/"

sequence_length = 100
step = 3

all_notes = []

for filename in os.listdir(midi_directory):
    if filename.endswith(".mid") or filename.endswith(".midi"):
        midi_file = os.path.join(midi_directory, filename)

        try:
            midi_data = pretty_midi.PrettyMIDI(midi_file)
            notes = []
            
            for instrument in midi_data.instruments:
                for note in instrument.notes:
                    notes.append(note.pitch)

            encoded_notes = convert_notes_to_numbers(notes)

            all_notes.extend(encoded_notes)
        except Exception as e:
            print("Error reading file. ",e)

num_features = len(set(all_notes))

X_train, y_train = training_data(all_notes, sequence_length)
