# Feed a random sample of inputs through a trained network to extract representative activations
# throughout the network. Useful for ensuring constraints imposed by the loss function are working,
# the network is free from dead spots, etc.
import os
import numpy as np
import pandas as pd
from keras import backend as K, models, utils
from tkinter import filedialog, Tk
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


matplotlib.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Verdana']})
BATCH_SIZE = 128
BATCHES_TO_SAMPLE = 10  # Number of batch-sized inputs of randomly selected data to feed the net.
HIST_BINS = 20
DPI = 200
SIZE_INCHES = (2, 1.5)  # Width, height.
Tk().withdraw()  # Hide default blank Tk window.
model_file = filedialog.askopenfilename(filetypes=[('HDF5 models', '.hdf5')], initialdir='.')
TIMESTEPS = 20 if '20steps' in model_file else \
            15 if '15steps' in model_file else \
            10 if '10steps' in model_file else \
            5 if '5steps' in model_file else -1
model_dir = model_file[:model_file.rfind('/') + 1]

try:
    os.mkdir(model_dir + 'visualizations')
except FileExistsError:
    pass

K.set_learning_phase(0)  # Set to test mode (vs. train; necessary for loading some models).

print('Loading model')
# Have to load structure and weights separately due to Keras issue (see #5916 on GitHub).
with open(model_dir + 'model_structure.json') as ifile:
    autoencoder = models.model_from_json(ifile.read())
autoencoder.load_weights(model_file)
utils.plot_model(autoencoder, show_shapes=True,
                 to_file=model_dir + 'visualizations/model_structure.png')

print('Loading data')  # Use only validation data for now.
val_X = np.load('bb/data/sequences_val-' + str(TIMESTEPS) + 'steps.npy')[:, :15]  # Use first 15s.

# Generate sample activations for every layer that has trainable weights.
for layer_index, layer in enumerate(autoencoder.layers):
    if len(layer.trainable_weights) > 0 or layer.name == 'embedding':
        print(layer.name)
        subgraph = K.function(inputs=[autoencoder.layers[0].input], outputs=[layer.output])
        # Randomly select some samples, construct batches, and feed.
        activations = []
        for batch in range(BATCHES_TO_SAMPLE):
            sample_i = np.random.choice(len(val_X), BATCH_SIZE, replace=False)
            sample_act = subgraph([val_X[sample_i]])[0]  # Feed forward.
            sample_act = np.reshape(sample_act, (BATCH_SIZE, -1))  # Batch flatten.
            activations.append(sample_act)
        z_len = len(activations[0][0])
        # Could save sample activations for further analysis if needed.
        df = pd.DataFrame(data=np.concatenate(activations),
                          columns=['z' + str(i) for i in range(z_len)])
        # Find a roughly nice layout for plotting.
        v_size = 1
        while z_len / v_size > v_size:
            v_size *= 2
        v_size = max(1, int(v_size / 2))
        h_size = int(z_len / v_size + .9999)

        # Create a figure with a subsample since there are so many.
        if z_len > 15:
            plt.figure(figsize=[5 * SIZE_INCHES[0], 3 * SIZE_INCHES[1]], dpi=DPI)
            cols = [c for c in df]
            for i in range(15):
                ax = plt.subplot(3, 5, i + 1)
                col = cols[int(i / 15 * len(cols))]
                plt.hist(df[col].values, HIST_BINS)
                plt.text(.96, .85, col, horizontalalignment='right', transform=ax.transAxes,
                         alpha=.6, fontdict={'size': 12, 'weight': 'bold'})
                if i % 5 == 0:
                    plt.ylabel('Frequency')
            plt.tight_layout(pad=0)
            plt.savefig(model_dir + 'visualizations/' + str(layer_index) + '-' +
                        layer.name + '-sample.png')
            plt.close()

        # Create a full figure if it doesn't seem too excessive.
        if z_len <= 160:
            plt.figure(figsize=[h_size * SIZE_INCHES[0], v_size * SIZE_INCHES[1]], dpi=DPI)
            for i, col in enumerate(df):
                ax = plt.subplot(v_size, h_size, i + 1)
                plt.hist(df[col].values, HIST_BINS)
                plt.text(.96, .85, col, horizontalalignment='right', transform=ax.transAxes,
                         alpha=.6, fontdict={'size': 12, 'weight': 'bold'})
                if i % h_size == 0:
                    plt.ylabel('Frequency')
            plt.tight_layout(pad=0)
            plt.savefig(model_dir + 'visualizations/' + str(layer_index) + '-' +
                        layer.name + '.png')
            plt.close()
        else:
            print('Too many features, refusing to make full plot (' + str(z_len) + ' found)')
