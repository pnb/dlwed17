# Try extract some simple features in the traditional method, to compare results.
import numpy as np
import pandas as pd
from collections import OrderedDict


TIMESTEPS = 20


print('Loading data')
df = pd.read_csv('bb/data/timeseries-1000.csv')
labeled_i = df[df.affect.notnull() | df.behavior.notnull()].index

print('Extracting features')
rows = []
for i in labeled_i:
    window = df.loc[i-TIMESTEPS+1:i]
    if len(window.participant_id.unique()) > 1 or \
            window.participant_id.iloc[0] != df.loc[i].participant_id:
        print('Skipping due to data spanning participants')
        continue
    rows.append(OrderedDict())
    rows[-1]['participant_id'] = df.loc[i].participant_id
    rows[-1]['orig_row_index'] = i
    rows[-1]['affect'] = df.loc[i].affect
    rows[-1]['behavior'] = df.loc[i].behavior
    rows[-1]['num_interactions_mean'] = window.num_interactions.mean()
    rows[-1]['view_causal_map_mean'] = window.action_ViewCausalMapAction.mean()
    rows[-1]['view_page_mean'] = window.action_ViewPageAction.mean()

print('Saving')
pd.DataFrame.from_records(rows).to_csv('bb/supervised/expert_feats_data.csv', index=False)
