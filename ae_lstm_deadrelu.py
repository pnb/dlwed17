# This version of the network produces dead ReLU neurons as an illustrative example.
# Lowering the learning rate to .01 results in all dead neurons.
import time
import numpy as np
import os
from keras import layers, models, callbacks, regularizers, optimizers


BATCH_SIZE = 128
TIMESTEPS = 20
TRAIN_TIMESTEPS = 15
EPOCHS = 5

# Create a unique run ID for this experiment that can be used, e.g., for TensorBoard logs.
run_id = 'bb_relu_' + str(TIMESTEPS) + 'steps_' + \
    time.strftime('%Y-%m-%d_%H.%M.%S', time.gmtime())
os.mkdir('bb/models/' + run_id)

print('Loading data')
train_X = np.load('bb/data/sequences_train-' + str(TIMESTEPS) + 'steps.npy')[:, :, [0, 2, 4]]
val_X = np.load('bb/data/sequences_val-' + str(TIMESTEPS) + 'steps.npy')[:, :, [0, 2, 4]]
if len(train_X) % BATCH_SIZE != 0 or len(val_X) % BATCH_SIZE != 0:
    raise ValueError('Data size incompatible with batch size (must be evenly divisible)')
num_inputs = len(train_X[0][0])
print('Number of inputs in loaded data: ' + str(num_inputs))


# Create graph structure.
input_placeholder = layers.Input(shape=[TRAIN_TIMESTEPS, num_inputs])
# Encoder.
encoded = layers.LSTM(64, return_sequences=True, activation='relu')(input_placeholder)
encoded = layers.LSTM(40, activation='relu')(encoded)
encoded = layers.Dense(3, activation='relu')(encoded)
encoded = layers.BatchNormalization(name='embedding')(encoded)
# Decoder.
decoded = layers.Dense(8, activation='relu')(encoded)
decoded = layers.Dense(16, activation='relu')(decoded)
decoded = layers.Dense((TIMESTEPS - TRAIN_TIMESTEPS) * num_inputs, activation='sigmoid')(decoded)
decoded = layers.Reshape([TIMESTEPS - TRAIN_TIMESTEPS, num_inputs])(decoded)

encoder = models.Model(inputs=input_placeholder, outputs=encoded)
autoencoder = models.Model(inputs=input_placeholder, outputs=decoded)
print(autoencoder.summary())
with open('bb/models/' + run_id + '/model_structure.json', mode='w') as ofile:
    ofile.write(autoencoder.to_json())

print('Compiling model')
opt = optimizers.RMSprop(lr=.0001)
autoencoder.compile(optimizer=opt, loss='mse')

autoencoder.fit(train_X[:, :TRAIN_TIMESTEPS], train_X[:, TRAIN_TIMESTEPS:],
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                validation_data=(val_X[:, :TRAIN_TIMESTEPS], val_X[:, TRAIN_TIMESTEPS:]),
                callbacks=[
                    # callbacks.TensorBoard(log_dir='tf_logs/' + run_id),
                    callbacks.ModelCheckpoint('bb/models/' + run_id +
                                              '/epoch{epoch:02d}-loss{val_loss:.3f}.hdf5'),
                    callbacks.ModelCheckpoint('bb/models/' + run_id + '/best.hdf5',
                                              save_best_only=True),
                ])
