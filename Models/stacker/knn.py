#import stacker
from sklearn.neighbors import KNeighborsRegressor
import splitter

import matplotlib.pyplot as plt

import time

import tkinter as tk
from PIL import ImageTk, Image

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


SEED = 42

X_train = splitter.X_train
Y_train = splitter.Y_train
X_test = splitter.X_test
Y_test = splitter.Y_test



reg = KNeighborsRegressor(n_neighbors = 5, n_jobs = -1)
reg.fit(X_train, Y_train)
predict = reg.predict(X_test)

r2_knn = r2_score(Y_test, predict)
mse_knn = mean_squared_error(Y_test, predict)
mae_knn = mean_absolute_error(Y_test, predict)

print(f"The R2_score is : {r2_knn}")
print(f"The Mean squared error is : {mse_knn}")
print(f"The Mean absolute error  is : {mae_knn}")

plt.scatter(predict, Y_test)
plt.savefig("knn.png")

print("The result of KNN")
time.sleep(2)

root = tk.Tk()

img = ImageTk.PhotoImage(Image.open("knn.png"))
label = tk.Label(root, image = img).pack()

root.after(5000, lambda: root.destroy())
root.mainloop()
