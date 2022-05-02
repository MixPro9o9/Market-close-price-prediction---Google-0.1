import pandas as pd
import numpy as np

data_df = pd.read_csv('GOOG.csv')

x = data_df.drop(['close', 'symbol', 'date'], axis=1).values
y = data_df['close'].values

from sklearn.model_selection import train_test_split

#split input_data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=10)

#train the model on the training set
from sklearn.linear_model import LinearRegression
ml = LinearRegression()
ml.fit(x_train, y_train)

#predict the test set results
y_pred = ml.predict(x_test)

#evaluate the model
from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))

#plot the result
import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.show()

#predicted values
pred_y_df = pd.DataFrame({'Actual Value' :y_test, 'Predicted Value' :y_pred, 'Difference' :y_test - y_pred})
print(pred_y_df[0:20])