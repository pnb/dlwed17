# Variational autoencoder (VAE) with K-L divergence warm-up at the batch level.
import numpy as np
import os
from keras import layers, models, callbacks, regularizers, optimizers, metrics, backend as K
from keras.layers import advanced_activations


KL_WARMUP_BATCHES = 1000
BATCH_SIZE = 128
TIMESTEPS = 20
TRAIN_TIMESTEPS = 15
EPOCHS = 5

# Create a unique run ID for this experiment that can be used, e.g., for TensorBoard logs.
run_id = 'bb_vae_' + str(TIMESTEPS) + 'steps_' + \
    time.strftime('%Y-%m-%d_%H.%M.%S', time.gmtime())
os.mkdir('bb/models/' + run_id)

print('Loading data')
train_X = np.load('bb/data/sequences_train-' + str(TIMESTEPS) + 'steps.npy')[:, :, [0, 2, 4]]
val_X = np.load('bb/data/sequences_val-' + str(TIMESTEPS) + 'steps.npy')[:, :, [0, 2, 4]]
if len(train_X) % BATCH_SIZE != 0 or len(val_X) % BATCH_SIZE != 0:
    raise ValueError('Data size incompatible with batch size (must be evenly divisible)')
num_inputs = len(train_X[0][0])
print('Number of inputs in loaded data: ' + str(num_inputs))


def sampling(args):
    mean, log_sigma = args
    epsilon = K.random_normal(shape=(128, 3), mean=0., stddev=1.0)
    return mean + K.exp(log_sigma) * epsilon


# Create graph structure.
input_placeholder = layers.Input(shape=[TRAIN_TIMESTEPS, num_inputs])
# Encoder.
encoded = layers.LSTM(64, return_sequences=True)(input_placeholder)
encoded = advanced_activations.ELU(alpha=.5)(encoded)
encoded = layers.LSTM(40)(encoded)
encoded = advanced_activations.ELU(alpha=.5)(encoded)

z_mean = layers.Dense(3)(encoded)
z_log_sigma = layers.Dense(3)(encoded)
encoded = layers.Lambda(sampling,
                        output_shape=(3,))([z_mean, z_log_sigma])

encoded = layers.BatchNormalization(name='embedding')(encoded)
# Decoder.
decoded = layers.Dense(8)(encoded)
decoded = advanced_activations.ELU(alpha=.5)(decoded)
decoded = layers.Dense(16)(decoded)
decoded = advanced_activations.ELU(alpha=.5)(decoded)
decoded = layers.Dense((TIMESTEPS - TRAIN_TIMESTEPS) * num_inputs, activation='sigmoid')(decoded)
decoded = layers.Reshape([TIMESTEPS - TRAIN_TIMESTEPS, num_inputs])(decoded)


kl_warmup_coeff = K.variable(0.0)


def vae_loss(x_true, x_predicted):
    x_true = K.batch_flatten(x_true)
    x_predicted = K.batch_flatten(x_predicted)
    reconstruction_loss = num_inputs * metrics.binary_crossentropy(x_true, x_predicted)
    kld_loss = -.5 * K.sum(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
    return reconstruction_loss + kl_warmup_coeff * kld_loss


def kl_batch_warmup(batch, logs):
    """
    Callback to increase the weight of the KL divergence term in the loss function gradually over
    the course of many batches.
    """
    if batch <= KL_WARMUP_BATCHES:  # Avoid some calls to K.get_value if possible.
        cur_val = K.get_value(kl_warmup_coeff)
        if cur_val < 1.0:
            K.set_value(kl_warmup_coeff, cur_val + (1.0 / KL_WARMUP_BATCHES))


encoder = models.Model(inputs=input_placeholder, outputs=encoded)
autoencoder = models.Model(inputs=input_placeholder, outputs=decoded)
print(autoencoder.summary())
with open('bb/models/' + run_id + '/model_structure.json', mode='w') as ofile:
    ofile.write(autoencoder.to_json())

print('Compiling model')
opt = optimizers.RMSprop(lr=.0001)
autoencoder.compile(optimizer=opt, loss=vae_loss)

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
                    callbacks.LambdaCallback(on_batch_begin=kl_batch_warmup),
                ])
