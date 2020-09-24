import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Importing the dataset
df = pd.read_csv('employee_attrition.csv')
df = pd.DataFrame(df)

# Splitting the data into two groups(existing and ex employees)
left = df.groupby('attrition')
left.mean()

# Subplots using seaborn/matplotlib
attributes = ['no_of_project','time_spend_company', 'work_accident', 'salary', 'attrition', 'promotion_last_5years']
fig = plt.subplots(figsize = (70,45))
for i, j in enumerate(attributes):
    plt.subplot(4, 2, i+1)
    plt.subplots_adjust(hspace = 0.1)
    sns.countplot(x =j, data = df, hue='attrition')
    plt.xticks(rotation=90)
    plt.title("No. of employee")
    plt.show()


## Pre-processing data
# Creating labelEncoder
le = preprocessing.LabelEncoder()

# Converting string labels into numbers.
df['salary']=le.fit_transform(df['salary'])
df['dept']=le.fit_transform(df['dept'])
df['attrition']=le.fit_transform(df['attrition'])

## Split train and test set
x = df[['satisfaction_level', 'last_evaluation', 'no_of_project',
       'average_monthly_hours', 'time_spend_company', 'work_accident',
       'promotion_last_5years', 'dept', 'salary']]

y = df['attrition']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state =42)

## Model building
rf_classifier = RandomForestClassifier()

rf = rf_classifier.fit(x_train, y_train)

y_pred = rf.predict(x_test)


# Evaluating the model performance
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Model accuracy is {accuracy}")

precision = metrics.precision_score(y_test, y_pred)
print(f'Model precision is {precision}')

recall = metrics.recall_score(y_test, y_pred)
print(f'Model recall is {recall}')


