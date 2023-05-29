# data-set Creation

import json

import pandas as pd
import ta
from ta.utils import dropna

# bufferframe = pd.read_csv('dataset.csv')
bufferframe = pd.read_csv('NSETickersFor-2023-04-27.csv')

#Step 0: prune unnecessary data
df =  bufferframe[['timestamp','volume', 'high', 'lastTradedPrice', 'close', 'low', 'open']]
# Clean NaN values
df = dropna(df)

# Calculate the simple moving average (SMA)
df['sma'] = ta.trend.sma_indicator(df['lastTradedPrice'], window=10)
# Generate labels based on simple directional change
df['Label'] = df['lastTradedPrice'].diff().apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)

# Drop rows with missing values (if any)
df.dropna(inplace=True) 

df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
print(df)

#create modified dataset 

# file = df.to_excel("output.xlsx",
#              sheet_name='Sheet_name_1')

file = df.to_excel("NSETickersFor-2023-04-27.inputDataset.xlsx",
             sheet_name='Sheet_name_1')