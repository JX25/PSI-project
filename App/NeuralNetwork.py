from pprint import pprint

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
import numpy as np
from App.main import facts, facts_vtp, votes_to_predict, votes
x_train = facts.astype(np.float64)
y_train = votes
x_predict = facts_vtp
y_predict = votes_to_predict
yt_pr = np.transpose(y_predict).astype(np.float64).tolist()
yt = np.transpose(y_train).astype(np.float64).tolist()

nn = MLPRegressor(activation='relu', solver='adam', hidden_layer_sizes=(500, 550, 300, 500, 230, 360), random_state=1)
nn.fit(x_train, yt)
train_mse = nn.predict(x_train)
test_mse = nn.predict(x_predict)
print('MSE training', mean_squared_error(train_mse, yt))
print('MSE testing', mean_squared_error(test_mse, yt_pr))

train_pred = nn.predict(x_train)
test_pred = nn.predict(x_predict)
pprint(y_predict)
pprint(test_pred)
pprint(train_pred)
