import tensorflow as tf
import numpy as np
import pandas as pd
import random
import os
import pretty_midi
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adamax
from pydub import AudioSegment

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Define the function to load and preprocess MIDI files
def load_midi_files(directory):
    notes = []
    for filename in os.listdir(directory):
        if filename.endswith(".mid"):
            midi_data = pretty_midi.PrettyMIDI(os.path.join(directory, filename))
            for instrument in midi_data.instruments:
                if instrument.is_drum:
                    continue
                for note in instrument.notes:
                    pitch = note.pitch
                    duration = note.end - note.start
                    notes.append((pitch, duration))
    return notes

# Load and preprocess the MIDI files
data_directory = "./dataset/"  # Replace with the path to your MIDI files directory
notes = load_midi_files(data_directory)

# Create a dictionary to map pitches to integers
pitchnames = sorted(set(note[0] for note in notes))
note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

# Prepare the training data
sequence_length = 100
n_vocab = len(pitchnames)
network_input = []
network_output = []

for i in range(len(notes) - sequence_length):
    sequence_in = notes[i:i + sequence_length]
    sequence_out = notes[i + sequence_length]
    network_input.append([note_to_int[note[0]] for note in sequence_in])
    network_output.append(note_to_int[sequence_out[0]])

n_patterns = len(network_input)
network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
network_input = network_input / float(n_vocab)
network_output = tf.keras.utils.to_categorical(network_output)

# Define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(256))
model.add(Dense(n_vocab, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adamax(learning_rate=0.001))

# Train the model
model.fit(network_input, network_output, epochs=100, batch_size=64)

# Generate music using the trained model
start = np.random.randint(0, len(network_input)-1)
pattern = network_input[start]
prediction_output = []

for _ in range(500):
    prediction_input = np.reshape(pattern, (1, len(pattern), 1))
    prediction_input = prediction_input / float(n_vocab)
    prediction = model.predict(prediction_input, verbose=0)
    index = np.argmax(prediction)
    result = pitchnames[index]
    prediction_output.append(result)
    pattern = np.append(pattern, index)
    pattern = pattern[1:]

# Convert the generated notes to a MIDI file
midi_stream = pretty_midi.PrettyMIDI()
piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
piano = pretty_midi.Instrument(program=piano_program)

for pitch in prediction_output:
    note = pretty_midi.Note(
        velocity=69, pitch=int(pitch), start=0, end=30
    )
    piano.notes.append(note)

midi_stream.instruments.append(piano)
output_path = "output.mid"  # Replace with the desired output path for the MIDI file
midi_stream.write(output_path)

# Convert the MIDI file to an audio file (MP3)
audio_path = "output.mp3"  # Replace with the desired output path for the audio file
AudioSegment.from_file(output_path).export(audio_path, format="mp3")
