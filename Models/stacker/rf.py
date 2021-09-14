from os import SEEK_DATA
import splitter

import warnings

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.ensemble import RandomForestRegressor
SEED = 42


X_train = splitter.X_train
Y_train = splitter.Y_train
X_test = splitter.X_test
Y_test = splitter.Y_test


regressor = RandomForestRegressor(random_state = SEED)
regressor.fit(X_train, Y_train)

predict = regressor.predict(X_test)


r2_stacker = r2_score(Y_test, predict)
mse_stacker = mean_squared_error(Y_test, predict)
mae_stacker = mean_absolute_error(Y_test, predict)

print(f"The R2_score is : {r2_stacker}")
print(f"The Mean squared error is : {mse_stacker}")
print(f"The Mean absolute error  is : {mae_stacker}")
