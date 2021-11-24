import glob
import os
import pickle
import numpy
from keras import Model
from keras.callbacks import ModelCheckpoint
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import BatchNormalization as BatchNorm
from keras.layers import Activation
from keras.layers import Bidirectional, CuDNNLSTM
from keras.layers import concatenate
from keras.layers import Input


def get_notes(midi_files_folder):
    """ Obter todas as notas e acordes dos arquivos midi no diretórios de teste """
    notes = []
    offsets = []
    duration = []

    for file in glob.glob(os.path.join(midi_files_folder, '*.mid')):
        midi = converter.parse(file)

        print("Parsing %s" % file)

        notes_to_parse = None

        try: # arquivo tem partes de instrumentos
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except: # arquivo tem notas em uma estrutura plana
            notes_to_parse = midi.flat.notes

        offsetBase = 0
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
            else:
                continue

            offsets.append(str(element.offset - offsetBase))
            duration.append(str(element.duration.quarterLength))
            offsetBase = element.offset

    with open(r'C:\Users\heffe\Desktop\TCC\notes', 'wb') as filepath:
            pickle.dump(notes, filepath)

    with open(r'C:\Users\heffe\Desktop\TCC\offsets', 'wb') as filepath:
            pickle.dump(offsets, filepath)

    with open(r'C:\Users\heffe\Desktop\TCC\duration', 'wb') as filepath:
            pickle.dump(duration, filepath)

    return notes, offsets, duration


def get_from_pickle(file_path):
    with open(file_path, "rb") as file:
        data = pickle.load(file)
    return data


def prepare_sequences(notes, n_vocab):
    """ Preparo as sequências de notas usadas pela Rede Neural """
    sequence_length = 100

    # obter todos os nomes de pitch
    pitchnames = sorted(set(item for item in notes))

     # crie um dicionário para mapear "pitches" para inteiros
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # criar sequências de entrada e as saídas correspondentes
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # remodelar a entrada em um formato compatível com camadas LSTM
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalizar entrada
    network_input = network_input / float(n_vocab)

    network_output = np_utils.to_categorical(network_output)

    return (network_input, network_output)


def create_network(network_input_notes, network_input_offsets, network_input_durations,
                   n_vocab_notes, n_vocab_offsets, n_vocab_durations):
    """ criar a estrutura da rede neural """
    # Branch of the network that considers notes
    inputNotesLayer = Input(shape=(network_input_notes.shape[1], network_input_notes.shape[2]))
    inputNotes = LSTM(
        256,
        input_shape=(network_input_notes.shape[1], network_input_notes.shape[2]),
        return_sequences=True
    )(inputNotesLayer)
    inputNotes = Dropout(0.2)(inputNotes)

    # Branch of the network that considers note offset
    inputOffsetsLayer = Input(shape=(network_input_offsets.shape[1], network_input_offsets.shape[2]))
    inputOffsets = LSTM(
        256,
        input_shape=(network_input_offsets.shape[1], network_input_offsets.shape[2]),
        return_sequences=True
    )(inputOffsetsLayer)
    inputOffsets = Dropout(0.2)(inputOffsets)

    # Branch of the network that considers note duration
    inputDurationsLayer = Input(shape=(network_input_durations.shape[1], network_input_durations.shape[2]))
    inputDurations = LSTM(
        256,
        input_shape=(network_input_durations.shape[1], network_input_durations.shape[2]),
        return_sequences=True
    )(inputDurationsLayer)
    # inputDurations = Dropout(0.3)(inputDurations)
    inputDurations = Dropout(0.2)(inputDurations)

    # Concatentate the three input networks together into one branch now
    inputs = concatenate([inputNotes, inputOffsets, inputDurations])

    # A cheeky LSTM to consider everything learnt from the three separate branches
    x = LSTM(512, return_sequences=True)(inputs)
    x = Dropout(0.3)(x)
    x = LSTM(512)(x)
    x = BatchNorm()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)

    # Time to split into three branches again...

    # Branch of the network that classifies the note
    outputNotes = Dense(128, activation='relu')(x)
    outputNotes = Dropout(0.3)(outputNotes)
    outputNotes = Dense(n_vocab_notes, activation='softmax', name="Note")(outputNotes)

    # Branch of the network that classifies the note offset
    outputOffsets = Dense(128, activation='relu')(x)
    outputOffsets = Dropout(0.3)(outputOffsets)
    outputOffsets = Dense(n_vocab_offsets, activation='softmax', name="Offset")(outputOffsets)

    # Branch of the network that classifies the note duration
    outputDurations = Dense(128, activation='relu')(x)
    outputDurations = Dropout(0.3)(outputDurations)
    outputDurations = Dense(n_vocab_durations, activation='softmax', name="Duration")(outputDurations)

    # Tell Keras what our inputs and outputs are
    model = Model(inputs=[inputNotesLayer, inputOffsetsLayer, inputDurationsLayer],
                  outputs=[outputNotes, outputOffsets, outputDurations])

    # Adam seems to be faster than RMSProp and learns better too
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # Useful to try RMSProp though

    # LOAD WEIGHTS HERE IF YOU WANT TO CONTINUE TRAINING!
    # model.load_weights(weights_name)

    return model


def train(model, network_input_notes, network_input_offsets, network_input_durations, network_output_notes, network_output_offsets, network_output_durations):
    """ treinar a rede neural """

    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit([network_input_notes, network_input_offsets, network_input_durations],
              [network_output_notes, network_output_offsets, network_output_durations], epochs=200, batch_size=64,
              callbacks=callbacks_list, verbose=1)


if __name__ == "__main__":
    notes, offsets, duration = get_notes(r"C:\Users\heffe\Desktop\TCC\test_1\midi_zelda")

    #notes, offsets, duration = get_notes_from_pickle(r"C:\Users\heffe\Desktop\RNN\Classical-Piano-Composer\data\notes")

    # get amount of pitch names
    n_vocab_notes = len(set(notes))
    n_vocab_offsets = len(set(offsets))
    n_vocab_durations = len(set(duration))

    network_input_notes, network_output_notes = prepare_sequences(notes, n_vocab_notes)
    network_input_offsets, network_output_offsets = prepare_sequences(offsets, n_vocab_offsets)
    network_input_durations, network_output_durations = prepare_sequences(duration, n_vocab_durations)

    model = create_network(network_input_notes,  network_input_offsets, network_input_durations,
                           n_vocab_notes, n_vocab_offsets, n_vocab_durations)

    train(model, network_input_notes, network_input_offsets,
          network_input_durations, network_output_notes,
          network_output_offsets, network_output_durations)