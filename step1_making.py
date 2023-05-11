#Librarires required
import pickle
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt #for graph
from sklearn.ensemble import VotingClassifier #used in esembling model
from sklearn.model_selection import StratifiedKFold #For k fold
from sklearn.ensemble import RandomForestClassifier #for random forest classifier
from sklearn.ensemble import GradientBoostingClassifier #FOR G.B. classifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report #for accuracy and classification report
from mpl_toolkits import mplot3d #For 3d plots

df = pd.read_csv("Exp_data.csv")
df.head()
x = df.drop('Condition', axis=1)
y = df.Condition

#**********************************
#        MODEL 1_1, 1_2 (TLC) 
#**********************************

df1 = df.drop(['Protein', 'Sugar'], axis=1)
x1 = df1.drop('Condition', axis = 1)
y1 = df1.Condition

#**********************************
#MODEL 1_1 - TLC USING RANDOM FOREST

#training
skf = StratifiedKFold(n_splits = 10)
model1_1 = RandomForestClassifier(n_estimators=10,  max_features = 'sqrt')

def training(train, test, fold_no):
  x_train = train.drop(['Condition'],axis=1)
  y_train = train.Condition
  x_test = test.drop(['Condition'],axis=1)
  y_test = test.Condition
  model1_1.fit(x_train, y_train)
  score = model1_1.score(x_test,y_test)
  #print('For Fold {} the test accuracy is {}'.format(str(fold_no),score))
  score1 = model1_1.score(x_train,y_train)
  print('For Fold {} the Train accuracy is {} and Test accuracy is: {}.'.format(str(fold_no),score1,score))
  y_pred_test = model1_1.predict(x_test)
  print(confusion_matrix(y_test, y_pred_test))
  print(classification_report(y_test, y_pred_test))

fold_no = 1
for train_index,test_index in skf.split(x1, y1):
  train = df1.iloc[train_index,:]
  test = df1.iloc[test_index,:]
  training(train, test, fold_no)
  fold_no += 1

#**********************************
#MODEL 1_2 - TLC USING GB

skf = StratifiedKFold(n_splits = 10)
model1_2 = GradientBoostingClassifier(n_estimators=20, learning_rate=0.1, max_features=2, max_depth=2, random_state=0)

def training(train, test, fold_no):
  x_train = train.drop(['Condition'],axis=1)
  y_train = train.Condition
  x_test = test.drop(['Condition'],axis=1)
  y_test = test.Condition
  model1_2.fit(x_train, y_train)
  score = model1_2.score(x_test,y_test)
  #print('For Fold {} the test accuracy is {}'.format(str(fold_no),score))
  score1 = model1_2.score(x_train,y_train)
  print('For Fold {} the Train accuracy is {} and Test accuracy is: {}.'.format(str(fold_no),score1,score))
  y_pred_test = model1_2.predict(x_test)
  print(confusion_matrix(y_test, y_pred_test))
  print(classification_report(y_test, y_pred_test))

fold_no = 1
for train_index,test_index in skf.split(x1, y1):
  train = df1.iloc[train_index,:]
  test = df1.iloc[test_index,:]
  training(train, test, fold_no)
  fold_no += 1

#**********************************
#      MODEL 2_1, 2_2 (PROTEIN) 
#**********************************

#Altering dataset - 
df2 = df.drop(['TLC', 'Sugar'], axis=1)
#dividing dataset - 
x2 = df2.drop('Condition', axis = 1)
y2 = df2.Condition

#**********************************
#MODEL 2_1 - PROTEIN USING RAN FOR

skf = StratifiedKFold(n_splits = 10)
model2_1 = RandomForestClassifier(n_estimators=10,  max_features = 'sqrt')

def training(train, test, fold_no):
  x_train = train.drop(['Condition'],axis=1)
  y_train = train.Condition
  x_test = test.drop(['Condition'],axis=1)
  y_test = test.Condition
  model2_1.fit(x_train, y_train)
  score = model2_1.score(x_test,y_test)
  #print('For Fold {} the test accuracy is {}'.format(str(fold_no),score))
  score1 = model2_1.score(x_train,y_train)
  print('For Fold {} the Train accuracy is {} and Test accuracy is: {}.'.format(str(fold_no),score1,score))
  y_pred_test = model2_1.predict(x_test)
  print(confusion_matrix(y_test, y_pred_test))
  print(classification_report(y_test, y_pred_test))

fold_no = 1
for train_index,test_index in skf.split(x2, y2):
  train = df2.iloc[train_index,:]
  test = df2.iloc[test_index,:]
  training(train, test, fold_no)
  fold_no += 1

#**********************************
#MODEL 2_2 - PROTEIN USING GRAD BOOSTING

skf = StratifiedKFold(n_splits = 10)
model2_2 = GradientBoostingClassifier(n_estimators=20, learning_rate=0.1, max_features=2, max_depth=2, random_state=0)

def training(train, test, fold_no):
  x_train = train.drop(['Condition'],axis=1)
  y_train = train.Condition
  x_test = test.drop(['Condition'],axis=1)
  y_test = test.Condition
  model2_2.fit(x_train, y_train)
  score = model2_2.score(x_test,y_test)
  #print('For Fold {} the test accuracy is {}'.format(str(fold_no),score))
  score1 = model2_2.score(x_train,y_train)
  print('For Fold {} the Train accuracy is {} and Test accuracy is: {}.'.format(str(fold_no),score1,score))
  y_pred_test = model2_2.predict(x_test)
  print(confusion_matrix(y_test, y_pred_test))
  print(classification_report(y_test, y_pred_test))

fold_no = 1
for train_index,test_index in skf.split(x2, y2):
  train = df2.iloc[train_index,:]
  test = df2.iloc[test_index,:]
  training(train, test, fold_no)
  fold_no += 1

#**********************************
#      MODEL 3_1, 3_2 (SUGAR) 
#**********************************

#Altering dataset - 
df3 = df.drop(['TLC', 'Protein'], axis=1)
#dividing dataset - 
x3 = df3.drop('Condition', axis = 1)
y3 = df3.Condition

#**********************************
#MODEL 3_1 - SUGAR USING RAN FOR

skf = StratifiedKFold(n_splits = 10)
model3_1 = RandomForestClassifier(n_estimators=10,  max_features = 'sqrt')

def training(train, test, fold_no):
  x_train = train.drop(['Condition'],axis=1)
  y_train = train.Condition
  x_test = test.drop(['Condition'],axis=1)
  y_test = test.Condition
  model3_1.fit(x_train, y_train)
  score = model3_1.score(x_test,y_test)
  #print('For Fold {} the test accuracy is {}'.format(str(fold_no),score))
  score1 = model3_1.score(x_train,y_train)
  print('For Fold {} the Train accuracy is {} and Test accuracy is: {}.'.format(str(fold_no),score1,score))
  y_pred_test = model3_1.predict(x_test)
  print(confusion_matrix(y_test, y_pred_test))
  print(classification_report(y_test, y_pred_test))

fold_no = 1
for train_index,test_index in skf.split(x3, y3):
  train = df3.iloc[train_index,:]
  test = df3.iloc[test_index,:]
  training(train, test, fold_no)
  fold_no += 1


#**********************************
#MODEL 3_2 - SUGAR USING GRAD BOOSTING

skf = StratifiedKFold(n_splits = 10)
model3_2 = GradientBoostingClassifier(n_estimators=20, learning_rate=0.1, max_features=2, max_depth=2, random_state=0)

def training(train, test, fold_no):
  x_train = train.drop(['Condition'],axis=1)
  y_train = train.Condition
  x_test = test.drop(['Condition'],axis=1)
  y_test = test.Condition
  model3_2.fit(x_train, y_train)
  score = model3_2.score(x_test,y_test)
  #print('For Fold {} the test accuracy is {}'.format(str(fold_no),score))
  score1 = model3_2.score(x_train,y_train)
  print('For Fold {} the Train accuracy is {} and Test accuracy is: {}.'.format(str(fold_no),score1,score))
  y_pred_test = model3_2.predict(x_test)
  print(confusion_matrix(y_test, y_pred_test))
  print(classification_report(y_test, y_pred_test))

fold_no = 1
for train_index,test_index in skf.split(x3, y3):
  train = df3.iloc[train_index,:]
  test = df3.iloc[test_index,:]
  training(train, test, fold_no)
  fold_no += 1


#**********************************
#      ENSEMBLE MODEL  
#**********************************

# Making the final model using voting classifier
model = VotingClassifier( estimators=[('m1_1', model1_1), ('m1_2', model1_2), ('m2_1', model2_1), ('m2_2', model2_2), ('m3_1', model3_1), ('m3_2', model3_2)], voting='hard')

skf = StratifiedKFold(n_splits = 10)

def training(train, test, fold_no):
  x_train = train.drop(['Condition'],axis=1)
  y_train = train.Condition
  x_test = test.drop(['Condition'],axis=1)
  y_test = test.Condition
  model.fit(x_train, y_train)
  score = model.score(x_test,y_test)
  #print('For Fold {} the test accuracy is {}'.format(str(fold_no),score))
  score1 = model.score(x_train,y_train)
  print('For Fold {} the Train accuracy is {} and Test accuracy is: {}.'.format(str(fold_no),score1,score))
  y_pred_test = model.predict(x_test)
  print(confusion_matrix(y_test, y_pred_test))
  print(classification_report(y_test, y_pred_test))

fold_no = 1
for train_index,test_index in skf.split(x, y):
  train = df.iloc[train_index,:]
  test = df.iloc[test_index,:]
  training(train, test, fold_no)
  fold_no += 1


#Part of api
pickle.dump(model, open("model1.pkl", "wb"))

