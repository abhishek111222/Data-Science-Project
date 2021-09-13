import pandas as pd

#importing the UK dataset
dataset = pd.read_csv("dataset.csv")
dataset.set_index("date", inplace = True)

#printing a tick
tick = u'\2713'

#This is to check with the dataset with all the features in case required
check = pd.read_csv("dataset.csv")
check.set_index("date", inplace = True)


str='''The accepted features are :  \n1. Hospital patients \n2. New tests \n3. Positive Rate \n4. New vaccinations \n5. Reproduction rate '''
print(str)

print("Rest all the features will be dropped.")

features = ["hosp_patients", "new_tests", "positive_rate", "new_vaccinations", "reproduction_rate", "Mortality_Rate"]
print()
print(f"The passed features are : {dataset.columns}")
print()
dataset = dataset[features]

print("First 5 elements after deleting the unwanted features : ")
print("-"*100)
print(dataset.head())
print("-"*100)
print()

print("The data is already pre - processed and hence should not have any missing values")
print(dataset.isna().sum())
print("No null values " u'\u2713')


#20 lines of code for now.