import pickle
from sklearn.metrics import accuracy_score
import pandas as pd
import ta
from ta.utils import dropna
from sklearn.model_selection import train_test_split
trainedModel = 'finalized_model.sav'

#step 0: load an untrained data 
df = pd.read_excel('NSETickersFor-2023-04-27.inputDataset.xlsx')

df = dropna(df)

# Calculate the simple moving average (SMA)
df['sma'] = ta.trend.sma_indicator(df['lastTradedPrice'], window=10)
# Generate labels based on simple directional change
df['Label'] = df['lastTradedPrice'].diff().apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)

# Drop rows with missing values (if any)
df.dropna(inplace=True) 
# normalize the dateTime stamp for each data
# df['timestamp'] = pd.to_datetime(df['timestamp']).map(pd.Timestamp.timestamp)

# Split the data into training and testing datasets
X = df.drop(['Label','timestamp'], axis=1)  # Input features
y = df['Label']  # Output labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# load the model from disk
loaded_model = pickle.load(open(trainedModel, 'rb'))
y_pred = loaded_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f" {X_test} ")
print(f"Accuracy: {accuracy}")