# Load a trained model and extract embeddings by feeding sequences through the net and saving the
# activations of the embedding layer.
import numpy as np
import pandas as pd
from keras import models
from keras import backend as K

BATCH_SIZE = 128
RUN_ID = 'bb_lstm_20steps_2017-06-14_19.57.18'
MODEL_FILE = 'best.hdf5'
TIMESTEPS = 20


K.set_learning_phase(0)  # Set to test mode (vs. train; necessary for loading some models).

print('Loading model')
# Have to load structure and weights separately due to Keras issue (see #5916 on GitHub).
with open('bb/models/' + RUN_ID + '/model_structure.json') as ifile:
    autoencoder = models.model_from_json(ifile.read())
autoencoder.load_weights('bb/models/' + RUN_ID + '/' + MODEL_FILE)

embedding_layer = [l for l in autoencoder.layers if l.name == 'embedding'][0]
# This seems a strange way to make the encoder but it works great.
encoder = K.function(inputs=[autoencoder.layers[0].input], outputs=[embedding_layer.output])

print('Loading data')
train_X = np.load('bb/data/sequences_train-' + str(TIMESTEPS) + 'steps.npy')[:, 5:, [0, 2, 4]]
val_X = np.load('bb/data/sequences_val-' + str(TIMESTEPS) + 'steps.npy')[:, 5:, [0, 2, 4]]
y_i = np.load('bb/data/sequences_y_i-' + str(TIMESTEPS) + 'steps.npy')

# Extract embeddings for all available sequences (indexed by y_i), and save the results with the
# original indices so that they can be matched up with labels.
print('Extracting training set embeddings')
embeddings = []
for i in range(0, len(train_X), BATCH_SIZE):
    print('%.1f%%' % (i / len(train_X) * 100), end='\r')
    embeddings.append(encoder([train_X[i:i + BATCH_SIZE]])[0])
embeddings = [np.concatenate(embeddings)[:len(y_i) - len(val_X)]]  # Trim off batch padding.
print('Extracting validation set embeddings')
for i in range(0, len(val_X), BATCH_SIZE):
    print('%.1f%%' % (i / len(val_X) * 100), end='\r')
    embeddings.append(encoder([val_X[i:i + BATCH_SIZE]])[0])

print('Concatenating')
df = pd.DataFrame(data=np.concatenate(embeddings),
                  columns=['z' + str(i) for i in range(len(embeddings[0][0]))])
print('Saving')
df.insert(0, 'original_index', y_i)
df.to_csv('bb/models/' + RUN_ID + '/' + MODEL_FILE.replace('.hdf5', '-predict.csv'), index=False)
