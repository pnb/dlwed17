# Loads embeddings that have been extracted from an autoencoder, and aligns them with BROMP labels
# that were present in the raw data, saving only the rows with labels. The resulting file can then
# be used in a supervised model for prediction.
import pandas as pd
import numpy as np


RUN_ID = 'bb_lstm_20steps_2017-06-14_19.57.18'
MODEL_FILE = 'best.hdf5'

print('Loading embeddings')
embeddings = pd.read_csv('bb/models/' + RUN_ID + '/' + MODEL_FILE.replace('.hdf5', '-predict.csv'))

print('Loading labels and metadata')
source_data = pd.read_csv('bb/data/timeseries-1000.csv')

print('Aligning with original data')
source_data = source_data[source_data.index.isin(embeddings.original_index)]
embeddings.insert(0, 'participant_id', source_data.participant_id.values)
embeddings.insert(1, 'window_start_ms', source_data.window_start_ms.values)
embeddings.insert(2, 'behavior', source_data.behavior.values)
embeddings.insert(3, 'affect', source_data.affect.values)

print('Selecting only rows with labels')
embeddings = embeddings[embeddings.behavior.notnull() | embeddings.affect.notnull()]

print('Saving')
embeddings.to_csv('bb/models/' + RUN_ID + '/' + MODEL_FILE.replace('.hdf5', '-predict-aligned.csv'),
                  index=False)
