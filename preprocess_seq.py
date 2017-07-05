# Turn timeseries data into sequences for training RNNs. This splits data into trainin/validation
# sets for autoencoders as well.
import numpy as np
import pandas as pd
import nn_util


TIMESTEPS = 20
BATCH_SIZE = 128
VALIDATION_SPLIT = .2


print('Loading data')
df = pd.read_csv('bb/data/timeseries-1000.csv')
print(str(len(df)) + ' rows')

print('Normalizing data')
numeric_cols = ['num_interactions',
                'action_CapabilityReservedAction',
                'action_ViewCausalMapAction',
                'action_CausalMapElementsMovedAction',
                'action_ViewPageAction',
                'action_RespondToContinuePromptAction',
                'action_RespondToMultipleChoicePromptAction'
                ]
df = nn_util.z_standardize(df, columns=numeric_cols, clip_magnitude=3)  # Winsorize.
df = nn_util.rescale(df, columns=numeric_cols)  # Rescale to [0, 1] range.
for col in numeric_cols:
    if df[col].std() < .1:
        print('WARNING! Column has low standard deviation: ' + col)
        print(df[col].describe())
for col in numeric_cols:
    df[col] = df[col].replace(np.nan, 0)

print('Making sequences')
X, y_i = nn_util.make_sequences(df, numeric_cols,
                                participant_id_col='participant_id',
                                sequence_len=TIMESTEPS,
                                verbose=True)

print('Splitting training/validation')
val_X = X[-int(len(X) / BATCH_SIZE * VALIDATION_SPLIT) * BATCH_SIZE:]
# Round up training set size, so there might be some overlap (< BATCH_SIZE) to ensure full coverage.
train_X = X[:int((len(X) - len(val_X)) / BATCH_SIZE + .999) * BATCH_SIZE]

print('Saving')
np.save('bb/data/sequences_train-' + str(TIMESTEPS) + 'steps.npy', train_X)
np.save('bb/data/sequences_val-' + str(TIMESTEPS) + 'steps.npy', val_X)
np.save('bb/data/sequences_y_i-' + str(TIMESTEPS) + 'steps.npy', y_i)
pd.DataFrame(data=numeric_cols).to_csv('bb/data/numeric_cols.txt')
