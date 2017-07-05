# Parse and concatenate BROMP logs so they can be easily matched with interaction logs.
import os
from collections import OrderedDict
import pandas as pd


BROMP_DIR = 'bb/data/raw_bromp/Corrected_anonymized_BROMP_Files/'

rows = []
for fname in os.listdir(BROMP_DIR):
    print('Processing ' + fname)
    with open(BROMP_DIR + fname) as f:
        header_key = []
        header = []
        data_key = []
        start_time_ms = 0
        for line in f:
            line = line.strip()  # Fix newline issues with cross-platform encodings.
            if line.startswith('FILE HEADER KEY:'):
                header_key = line.split(',')[1:]
            elif not header and header_key:
                header = {k: v for k, v in zip(header_key, line.split(','))}
                if header['ntptimestamp_ms'] == 'NTPFAIL':
                    print('Skipping BROMP file due to missing NTP start time')
                    break
                start_time_ms = int(header['ntptimestamp_ms'][-13:])
            elif line.startswith('FILE DATA KEY:'):
                data_key = line.split(',')[1:]
            else:
                rows.append(OrderedDict())
                for k, v in zip(data_key, line.split(',')):
                    rows[-1][k] = v
                rows[-1]['timestamp_ms'] = start_time_ms + int(rows[-1]['msoffsetfromstart'])
                rows[-1]['classname'] = header['classname']
                rows[-1]['bromp_username'] = header['username']
                rows[-1]['bromp_file'] = fname

print('Saving')
pd.DataFrame.from_records(rows).to_csv('bb/data/bromp_processed.csv', index=False)
