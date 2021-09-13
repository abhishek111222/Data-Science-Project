import splitter
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, StackingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import matplotlib.pyplot as plt
from PIL import Image
import subprocess
from sklearn.svm import SVR
import time
import cv2
from sklearn.linear_model import SGDRegressor
import warnings
import tkinter as tk
from PIL import ImageTk, Image
warnings.filterwarnings("ignore")
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

SEED = 42

X_train = splitter.X_train
Y_train = splitter.Y_train
X_test = splitter.X_test
Y_test = splitter.Y_test

estimators = [
    ("knn", KNeighborsRegressor(n_neighbors = 10)),
    ("rf", RandomForestRegressor(random_state = SEED, n_estimators = 100)),
    #("rf1", RandomForestRegressor(random_state = SEED, n_estimators = 80)),
    #("bag", BaggingRegressor(random_state = SEED, base_estimator = SVR())),
    ("sgd", SGDRegressor()),
  # ("xgb1", xgb.XGBRegressor(random_state = SEED)),
   #("rf3", RandomForestRegressor(random_state = SEED)),
   ("xgb", xgb.XGBRegressor(random_state = SEED)),
]

reg = StackingRegressor(
    estimators = estimators,
    final_estimator = BaggingRegressor(base_estimator = SVR(), random_state = SEED)
)

reg.fit(X_train, Y_train)
predict = reg.predict(X_test)

r2_stacker = r2_score(Y_test, predict)
mse_stacker = mean_squared_error(Y_test, predict)
mae_stacker = mean_absolute_error(Y_test, predict)

print(f"The R2_score is : {r2_stacker}")
print(f"The Mean squared error is : {mse_stacker}")
print(f"The Mean absolute error  is : {mae_stacker}")
time.sleep(5)


plt.scatter(predict, Y_test)
plt.savefig("1.png")

root = tk.Tk()
root.title("Result of Stacker")
root.geometry("4500x3000")  

img = ImageTk.PhotoImage(Image.open("1.png"))
label = tk.Label(root, image = img).pack()

root.after(5000, lambda: root.destroy())
root.mainloop()


#50 lines of code for now