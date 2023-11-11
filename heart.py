import numpy as np
import pandas as pd


from sklearn.metrics import accuracy_score
# loading the csv data to a Pandas DataFrame
heart_data = pd.read_csv("E:/project/multiple disease/heart.csv")

##Data Collection and Processing

# print first 5 rows of the dataset
heart_data.head()

# print last 5 rows of the dataset
heart_data.tail()

# number of rows and columns in the dataset
heart_data.shape

# getting some info about the data
heart_data.info()

# checking for missing values
heart_data.isnull().sum()

# statistical measures about the data
heart_data.describe()

# checking the distribution of Target Variable
heart_data['target'].value_counts()

heart_data.columns

cols=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

import seaborn as sns
import matplotlib.pyplot as plt

for i in cols:
    sns.boxplot(heart_data[i]); plt.show()

IQR=heart_data['chol'].quantile(0.75)-heart_data['chol'].quantile(0.25)
IQR
lower_limit=heart_data['chol'].quantile(0.25)-IQR*1.5
lower_limit
upper_limit=heart_data['chol'].quantile(0.75)+IQR*1.5
upper_limit

###### 1.Remove (Lets Trim the dataset)
####Trimming Technique###
###Lets Flag the outliers in the Dataset
Outliers_heart_data=np.where(heart_data['chol'] > upper_limit,True,np.where(heart_data['chol'] < lower_limit,True,False))
heart_data_trimmed=heart_data.loc[~(Outliers_heart_data),]
heart_data_trimmed
sum(Outliers_heart_data)
heart_data.shape,heart_data_trimmed.shape
#####Lets Explore the Outliers in the trimmed Datasets
sns.boxplot(heart_data_trimmed.chol)
plt.title('boxplot')

### Here we see that there are Outliers
##### 2.Replace (Lets trim the Datasets)
###Lets Replace the Outliers by the Maximum and minimum limit
heart_data['heart_data_Replaced']=pd.DataFrame(np.where(heart_data['chol'] > upper_limit,upper_limit,np.where(heart_data['chol'] < lower_limit,lower_limit,heart_data['chol'])))
sns.boxplot(heart_data.heart_data_Replaced)
plt.title('Boxplot')
plt.show()

##### 3.winsorization
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['age', 'sex', 'cp', 'trestbps', 'chol', 'restecg', 'thalach',
                                 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'])
heart_data_t=winsor.fit_transform(heart_data[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']])
heart_data_t

#we can inspect the minimum caps and maximum caps
#winsor.left_tail_caps_,winsor.right_tail_caps_
#let see boxplot
sns.boxplot(heart_data_t.trestbps)
sns.boxplot(heart_data_t.chol)
sns.boxplot(heart_data_t.fbs)
sns.boxplot(heart_data_t.thalach)
sns.boxplot(heart_data_t.oldpeak)
sns.boxplot(heart_data_t.ca)
sns.boxplot(heart_data_t.thal)
plt.title('boxplot')
plt.show()

##Splitting the Features and Target

X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

print(X)

print(Y)

###Splitting the Data into Training data & Test Data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

##Model Training
##Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

# training the LogisticRegression model with Training data
model.fit(X_train, Y_train)

##Model Evaluation
##Accuracy Score
# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on Training data : ', training_data_accuracy)

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on Test data : ', test_data_accuracy)

##Building a Predictive System

input_data = (62,0,0,140,268,0,0,160,0,3.6,0,2,2)

# change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]== 0):
  print('The Person does not have a Heart Disease')
else:
  print('The Person has Heart Disease')

##Saving the trained model
import pickle
filename = 'heart_disease_model.sav'
pickle.dump(model, open(filename, 'wb'))

# loading the saved model
loaded_model = pickle.load(open('heart_disease_model.sav', 'rb'))

for column in X.columns:
  print(column)



