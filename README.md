# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas
2. Import Decision tree classifier
3. Fit the data in the model
4. Find the accuracy score

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: VENKATA MOHAN N
RegisterNumber: 212224230298
*/
```
```
import pandas as pd
data=pd.read_csv("Employee.csv")
print("data.head():")
data.head()
```
```
print("data.info():")
data.info()
```
```
print("isnull() and sum():")
data.isnull().sum()
```
```
print("data value counts():")
data["left"].value_counts()
```
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
```
```
print("data.head() for Salary:")
data["salary"]=le.fit_transform(data["salary"])
data.head()
```
```
print("x.head():")
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
```
```
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
```
```
print("Accuracy value:")
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
```
print("Data Prediction:")
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

```
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plot_tree(dt, feature_names=x.columns, class_names=['salary', 'left'], filled=True)
plt.show()

```
## Output:
![image](https://github.com/user-attachments/assets/0a4fa5db-4d79-4ebd-9926-f4d1ee83d1e3)

![image](https://github.com/user-attachments/assets/ea8d6d09-3cc4-4875-ade1-46c9aa59e5c0)

![image](https://github.com/user-attachments/assets/9b850bb3-7c7e-4775-b2e3-789789375431)

![image](https://github.com/user-attachments/assets/7b31af4d-1ce9-4ff7-b45c-938ce231cb83)

![image](https://github.com/user-attachments/assets/71654c7d-2c86-4613-8d03-a010cb8a2b11)

![image](https://github.com/user-attachments/assets/5442db03-d63b-404e-ab1f-d9552a1a6a83)

![image](https://github.com/user-attachments/assets/727ed6aa-319b-4a29-8147-e094729d5549)

![image](https://github.com/user-attachments/assets/0e78318f-49be-400e-8e9d-3d4360dc9cc3)

![image](https://github.com/user-attachments/assets/4d27bab8-fef9-46a9-869c-23296ad3caac)









## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
