# Contains utility functions for neural network code, with documentation for each.
import numpy as np


def z_standardize(train_pandas_df, test_pandas_df=None, columns=None, clip_magnitude=None):
    """
    Shift and rescale data to have mean = 0 and standard deviation = 1, based on mean/sd computed
    from training data only (to prevent train/test set leakage). Missing values are ignored.

    :param train_pandas_df: training data as a pandas DataFrame
    :param test_pandas_df: testing data as a pandas DataFrame (optional)
    :param columns: list of columns to standardize (typically in + out columns in the network); if
                    none, then all columns will be standardized
    :param clip_magnitude: clip values beyond clip_magnitude standard deviations (i.e. Winsorize)
    :returns: tuple of (train, test) DataFrame copies with "columns" standardized, or only train if
              test_pandas_df was not specified
    """
    train = train_pandas_df.copy()
    test = test_pandas_df.copy() if test_pandas_df is not None else None
    for col in columns if columns is not None else train_pandas_df:
        train_m = train[col].mean()
        train_sd = train[col].std()
        train[col] = (train[col] - train_m) / train_sd
        if test is not None:
            test[col] = (test[col] - train_m) / train_sd
        if clip_magnitude is not None:
            train[col] = [min(max(v, -clip_magnitude), clip_magnitude) for v in train[col]]
            if test is not None:
                test[col] = [min(max(v, -clip_magnitude), clip_magnitude) for v in test[col]]
    return (train, test) if test is not None else train


def rescale(train_pandas_df, test_pandas_df=None, columns=None, new_min=0, new_max=1):
    """
    Shift and rescale data to have min = new_min and max = new_max, based on min and max computed
    from training data only (to prevent train/test set leakage). Missing values are ignored.

    :param train_pandas_df: training data as a pandas DataFrame
    :param test_pandas_df: testing data as a pandas DataFrame (optional)
    :param columns: list of columns to standardize (typically in + out columns in the network); if
                    none, then all columns will be standardized
    :param new_min: new minimum in the rescaled output
    :param new_max: new maximum in the rescaled output
    :returns: tuple of (train, test) DataFrame copies with "columns" rescaled, or only train if
              test_pandas_df was not specified
    """
    train = train_pandas_df.copy()
    test = test_pandas_df.copy() if test_pandas_df is not None else None
    for col in columns if columns is not None else train_pandas_df:
        train_min = train[col].min()
        train_range = train[col].max() - train_min
        train[col] = (train[col] - train_min) / train_range * (new_max - new_min) + new_min
        if test is not None:
            test[col] = (test[col] - train_min) / train_range * (new_max - new_min) + new_min
    return (train, test) if test is not None else train


def make_sequences(pandas_df,
                   in_columns,
                   participant_id_col=None,
                   sequence_len=10,
                   min_valid_prop=.7,
                   missing_fill=0,
                   overlap=1,
                   verbose=False):
    """
    Create sequences of data of a specific length using a sliding window that moves by "overlap".
    Sequences are necessary for training sequential models like LSTMs/GRUs. Also finds the indices
    of data points that can possibly be predicted, so targets (i.e. y) can be extracted.

    :param pandas_df: pandas DataFrame object with sequential data
    :param in_columns: list of columns to include in the resulting input sequences (i.e., X)
    :param participant_id_col: if specified, sequences will not overlap different participant IDs;
                               i.e., no sequence will include data from more than one participant
    :param sequence_len: number of rows to include in each sequence
    :param min_valid_prop: minimum proportion of non-missing data points per sequence; the sequence
                           will not be included in the result if this constraint is not met
    :param missing_fill: replace missing values with this (iff min_valid_prop is satisfied)
    :param overlap: number of rows to move the sliding window forward by (usually 1)
    :param verbose: print the number of sequences created every 1000 sequences
    :returns: tuple of (ndarray of sequences [i.e. inputs], list of target indices)
    """
    seqs = []
    indices = []
    for i in range(sequence_len, len(pandas_df) + 1, overlap):
        seq = pandas_df.iloc[i - sequence_len:i][in_columns]
        if participant_id_col and \
                len(pandas_df.iloc[i - sequence_len:i][participant_id_col].unique()) > 1:
            continue  # Cannot have sequences spanning multiple participant IDs.
        if seq.count().sum() / float(sequence_len * len(in_columns)) < min_valid_prop:
            continue  # Not enough valid data in this sequence.
        seqs.append(seq.fillna(missing_fill).values)
        indices.append(i - 1)
        if verbose and i / overlap % 1000 == 0:
            print('%.1f%%' % (i / len(pandas_df) * 100), end='\r')
    return np.array(seqs), indices


if __name__ == '__main__':
    # Do some testing.
    import pandas as pd
    df = pd.DataFrame.from_records([{'pid': 'p1', 'a': 1, 'b': 2, 'c': 'x'},
                                    {'pid': 'p1', 'a': 4.5},
                                    {'pid': 'p2', 'a': 2, 'b': 1, 'c': 'y'},
                                    {'pid': 'p2', 'a': 3, 'b': 0, 'c': 'x'},
                                    {'pid': 'p2', 'a': 4, 'b': 0, 'c': 'z'}])
    print(df)
    print('Sequences with strict min_valid_prop requirement:')
    print(make_sequences(df, ['a', 'b'], sequence_len=3, min_valid_prop=.9))
    print('Sequences with less strict min valid data:')
    print(make_sequences(df, ['a', 'b'], sequence_len=3, min_valid_prop=.5))
    print('Sequences with string column:')
    print(make_sequences(df, ['c'], sequence_len=3, min_valid_prop=.5, missing_fill='_'))
    print('Sequences bounded by participant id:')
    print(make_sequences(df, ['a'], participant_id_col='pid', sequence_len=2))

    print('Standardized with pid=p2 as train, pid=p1 as test:')
    a, b = z_standardize(df[df.pid == 'p2'], df[df.pid == 'p1'], ['a', 'b'])
    print(a)
    print(b)

    print('Rescaled to [-1, 1] with pid=p2 as train, pid=p1 as test:')
    a, b = rescale(df[df.pid == 'p2'], df[df.pid == 'p1'], ['a', 'b'], new_min=-1, new_max=1)
    print(a)
    print(b)
