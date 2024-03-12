# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and
.duplicated() function respectively
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required
modules from sklearn.
7. Apply new unknown values

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SHAKTHI SUNDAR K
RegisterNumber:  212222040152
*/

import pandas as pd
data=pd.read_csv("/content/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy Score:",accuracy)

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
print("\nConfusion Matrix:\n",confusion)


from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print("\nClassification Report:\n",classification_report1)

from sklearn import metrics
cm_display=metrics.ConfusionMatrixDisplay(confusion_matrix=confusion,display_labels=[True,False])
cm_display.plot()
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])


```

## Output:
### DATA:
![image](https://github.com/ShakthiSundar-K/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128116143/7f9a99c9-ae03-4263-8b1a-6828091571dd)
### ENCODED DATA:
![image](https://github.com/ShakthiSundar-K/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128116143/a58bb060-ef27-442d-a210-ee7f4a07da0d)
### NULL FUNCTION:
![image](https://github.com/ShakthiSundar-K/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128116143/d4476d4a-1b09-4ff3-9ab2-793ce7999290)
### DATA DUPLICATE:
![image](https://github.com/ShakthiSundar-K/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128116143/93616be4-90da-4e40-91a0-f738028a7940)
### ACCURACY:
![image](https://github.com/ShakthiSundar-K/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128116143/3772c63c-9fe1-400c-b799-d408d42d48d8)
### CONFUSION MATRIX:
![image](https://github.com/ShakthiSundar-K/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128116143/ab583420-e4ef-4f6d-96a4-0fd68c38dbe8)
### CONFUSION REPORT:
![image](https://github.com/ShakthiSundar-K/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128116143/b8221c69-0d65-4a67-8aa0-7b21a1c292e1)
### PREDICTION OF LR:
![image](https://github.com/ShakthiSundar-K/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128116143/1c865b48-0734-466a-8468-0b738585eaaa)
### GRAPH:
![image](https://github.com/ShakthiSundar-K/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128116143/fa1e3948-186b-4430-bd3c-e56a25ab85e0)










## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
