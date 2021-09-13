import data
from sklearn.preprocessing import StandardScaler

dataset = data.dataset

print()
print("The dataset will be divided into train and test. It will then be Scaled.")

#splitting the dataset
train = dataset[ : 400]
test = dataset[400 :] 

print()
print(f"The length of the train set is : {len(train)}")
print(f"The length of the test set is : {len(test)}")
print()


print("The Mortality Rate is set as the Target variable and the other features as the input")

#dividing the dataset in the target variable and the input
X_train = train.iloc[ : , : -1]
X_test = test.iloc[ : , : -1]
Y_train = train.iloc[ : , -1 : ]
Y_test = test.iloc[ : , -1 : ]


#scaling the dataset
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
Y_train = sc.fit_transform(Y_train)
Y_test = sc.fit_transform(Y_test)

print()
print("The data is scaled successfully.")
print()



#20 lines of code for now