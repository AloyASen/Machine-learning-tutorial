import pandas as pd
import quandl 

df = quandl.get('WIKI/GOOGL')
# remove the unvanted features 
df= df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
# add some extra features
df['HL_PCT']=  (df['Adj. High']- df['Adj. Close'])/df['Adj. Close']  *100
df['PCT_change']=  (df['Adj. Close']- df['Adj. Open'])/df['Adj. Open']  *100

df= df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
print(df.head())
