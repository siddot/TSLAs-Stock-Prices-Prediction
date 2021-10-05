import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

tsla = pd.read_csv('TSLA.csv')
tsla.head()

plt.title('Tesla')
plt.ylabel('Close Price USD ($)')
plt.xlabel('Days')
plt.plot(netflix['Close'])

close_price_df = tsla[['Close']]
close_price_df.head() 

future_days = 25
close_price_df['Predictions'] = close_price_df[['Close']].shift(-future_days)
close_price_df.head()

X = np.array(close_price_df.drop(["Predictions"], 1))[:-future_days]
y = np.array(close_price_df[['Predictions']])[:-future_days]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lin = LinearRegression()
lin.fit(X_train, y_train)

dec_tree = DecisionTreeRegressor()
dec_tree.fit(X_train, y_train)

x_future = close_price_df.drop(['Predictions'], 1)[:-future_days]
x_future = x_future.tail(future_days)
x_future = np.array(x_future)

pred = dec_tree.predict(x_future)
print(pred)

valid = df[X.shape[0]:]
valid['Predictions'] = pred
plt.figure(figsize=(24,12))

plt.title('Preds')
plt.xlabel('Days')
plt.ylabel('Close Price USD ($)') 
plt.plot(df[['Close']])
plt.plot(valid[['Predictions']])
plt.legend(['Orig', 'Val', 'Predictions'])
plt.show()
