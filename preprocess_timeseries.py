# Process action-level data from Betty's Brain into timeseries data where every step in the
# timeseries represents the same amount of time. This avoids issues of padding in RNNs, and allows
# the network to learn things about timing without an explicit timing column.
import os
import numpy as np
from collections import OrderedDict
import pandas as pd


WINDOW_SIZE = 1000  # In milliseconds for these data.
BROMP_MATCH_TOLERANCE = 10000  # Milliseconds.
MAX_IDLE_WINDOWS = 50  # After this, jump ahead to whatever the next timestamp is.


print('Loading data')
df = pd.read_csv(
    'bb/data/detailed-action.csv')
print('Loaded ' + str(len(df)) + ' rows of raw data')

print('Dummy coding data')
feature_cols = []
for value in df.action_name.unique():
    value = value[value.rfind('.') + 1:]
    df['action_' + value] = [1 if v.endswith(value) else 0 for v in df.action_name]
    feature_cols.append('action_' + value)

print('Creating timeseries data')
rows = []
for pid in sorted(df.subject_id.unique()):
    print('Processing participant ' + pid)
    pid_df = df[df.subject_id == pid]
    idle_windows = 0
    win_start_ms = pid_df.iloc[0].start_time
    while win_start_ms < pid_df.iloc[-1].end_time:
        win_end_ms = win_start_ms + WINDOW_SIZE - 1
        win_df = pid_df[(pid_df.start_time >= win_start_ms) &
                        (pid_df.start_time <= win_end_ms) |
                        ((pid_df.start_time < win_start_ms) & (pid_df.end_time >= win_end_ms))]
        if len(win_df) == 0:
            idle_windows += 1
        else:
            idle_windows = 0

        if idle_windows < MAX_IDLE_WINDOWS:
            if len(rows) % 1000 == 0:
                print('%.1f%%' % (100 * (win_start_ms - pid_df.iloc[0].start_time) /
                                  (pid_df.iloc[-1].end_time - pid_df.iloc[0].start_time)), end='\r')
            rows.append(OrderedDict())
            rows[-1]['participant_id'] = pid
            rows[-1]['window_start_ms'] = win_start_ms
            rows[-1]['window_end_ms'] = win_end_ms
            rows[-1]['num_interactions'] = len(win_df)
            for col in feature_cols:
                rows[-1][col] = win_df[col].mean()
            win_start_ms += WINDOW_SIZE
        else:
            # Jump ahead to the next non-idle section of data.
            win_start_ms = pid_df[pid_df.start_time > win_start_ms].iloc[0].start_time

print('Converting timeseries rows to DataFrame')
df = pd.DataFrame.from_records(rows)
df.to_csv('bb/data/timeseries-' + str(WINDOW_SIZE) + '.csv', index=False)  # Save now just in case.

print('Aligning BROMP')
bromp = pd.read_csv('bb/data/bromp_processed.csv')
df.insert(1, 'behavior', np.nan)
df.insert(2, 'affect', np.nan)
for i, row in bromp.iterrows():
    if i % 10 == 0:
        print('%.1f%%' % (i * 100 / len(bromp)), end='\r')
    matches = df[(df.participant_id == row.studentid) &
                 (df.window_start_ms <= row.timestamp_ms) &
                 (df.window_start_ms > row.timestamp_ms - BROMP_MATCH_TOLERANCE)].index
    if len(matches) == 0:
        print('Could not match BROMP row ' + str(i))
    else:
        df.loc[matches[-1], 'behavior'] = row.behavior
        df.loc[matches[-1], 'affect'] = row.affect
print('Matched ' + str(len(df[df.affect.notnull()])) + ' of ' + str(len(bromp)))

print('Saving')
df.to_csv('bb/data/timeseries-' + str(WINDOW_SIZE) + '.csv', index=False)
