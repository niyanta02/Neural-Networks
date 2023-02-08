# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 22:34:35 2022

@author: admin
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
import joblib

class NumericalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        X.drop(columns=['ADMIT TERM CODE', 'ADMIT YEAR', 'ID 2'], inplace=True)

        X.drop(columns=['FUTURE TERM ENROL'], inplace=True)

        return X[self.cols]
    
    
    
class CategoricalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['MAILING COUNTRY NAME'].fillna('Canada', inplace=True)
        X['PREV EDU CRED LEVEL NAME'].fillna(X['PREV EDU CRED LEVEL NAME'].mode()[0], inplace=True)
        X['APPLICANT CATEGORY NAME'].fillna(X['APPLICANT CATEGORY NAME'].mode()[0], inplace=True)
        X['AGE GROUP LONG NAME'].fillna(X['AGE GROUP LONG NAME'].mode()[0], inplace=True)

        X['CURRENT STAY STATUS'] = X['CURRENT STAY STATUS'].apply(change_STAY_STATUS)

        return X[self.cols]
    
def change_STAY_STATUS(row):
    if 'Left College' in row:
        return 'Left College'
    if 'Completed' in row:
        return 'Completed'
    if 'Graduated' in row:
        return 'Graduated'

dataset = pd.read_excel('F:/sem3/nn/HYPE Dataset.xlsx', sheet_name='HYPE-Retention')
print(f'the shape of data = {dataset.shape}')

dataset.drop(dataset[dataset['SUCCESS LEVEL'] == 'In Progress'].index, inplace=True)
dataset['STUDENT TYPE GROUP NAME'].unique()
dataset.head()

dataset.columns
'''
dataset.drop(columns=['HS AVERAGE MARKS', 'HS AVERAGE GRADE', 'APPL FIRST LANGUAGE DESC', ], inplace=True)
dataset['MAILING COUNTRY NAME'].fillna('Canada', inplace=True)
dataset['ENGLISH TEST SCORE'].fillna(dataset['ENGLISH TEST SCORE'].mean(), inplace=True)
dataset['PREV EDU CRED LEVEL NAME'].fillna(dataset['PREV EDU CRED LEVEL NAME'].mode()[0], inplace=True)
dataset['APPLICANT CATEGORY NAME'].fillna(dataset['PREV EDU CRED LEVEL NAME'].mode()[0], inplace=True)
dataset['AGE GROUP LONG NAME'].fillna(dataset['AGE GROUP LONG NAME'].mode()[0], inplace=True)


dataset.drop(dataset[dataset['ADMIT TERM CODE'].isna() == True].index, inplace=True)
#dataset['ADMIT TERM CODE'] = dataset['ADMIT TERM CODE'].astype(float)

dataset['ADMIT TERM CODE'].fillna(dataset['ADMIT TERM CODE'].mode(), inplace=True)
dataset['ADMIT YEAR'] = dataset['ADMIT TERM CODE'].astype(str).str[:4]

dataset['ADMIT MONTH']  = dataset['ADMIT TERM CODE'].astype(str).str[4:]
dataset['ADMIT MONTH'] =  dataset['ADMIT MONTH'].astype(int)

dataset['EXPECTED GRAD YEAR'] = dataset['EXPECTED GRAD TERM CODE'].astype(str).str[:4]
dataset['EXPECTED GRAD YEAR'] =  dataset['EXPECTED GRAD YEAR'].astype(int)

dataset['EXPECTED GRAD MONTH']  = dataset['EXPECTED GRAD TERM CODE'].astype(str).str[4:]
dataset['EXPECTED GRAD MONTH'] =  dataset['EXPECTED GRAD MONTH'].astype(int)

dataset = dataset.groupby(["ID 2"]).apply(lambda x: x.sort_values(["ADMIT YEAR"], ascending = False)).drop_duplicates('ID 2', keep='first').reset_index(drop=True)
dataset.drop(columns = ['APPL EDUC INST TYPE NAME','MAILING POSTAL CODE GROUP 3','MAILING PROVINCE NAME','MAILING POSTAL CODE','STUDENT TYPE GROUP NAME','STUDENT TYPE NAME','PROGRAM LONG NAME','INTAKE TERM CODE','ADMIT TERM CODE','EXPECTED GRAD TERM CODE', 'ID 2'], inplace=True)
dataset.shape
'''
dataset[['Term1', 'Term2', 'Term3', 'Term4', 'Term6', 'Term5', 'Term7', 'Term8', 'Term9', 'Term10']] = dataset['FUTURE TERM ENROL'].str.split('-',  expand=True)
dataset['Term1'] = dataset['Term1'].astype(int)
dataset['Term2'] = dataset['Term2'].astype(int)
dataset['Term3'] = dataset['Term3'].astype(int)
dataset['Term4'] = dataset['Term4'].astype(int)
dataset['Term6'] = dataset['Term5'].astype(int)
dataset['Term5'] = dataset['Term6'].astype(int)
dataset['Term7'] = dataset['Term7'].astype(int)
dataset['Term8'] = dataset['Term8'].astype(int)
dataset['Term9'] = dataset['Term9'].astype(int)
dataset['Term10'] = dataset['Term10'].astype(int)

#dataset.drop(columns=['FUTURE TERM ENROL'], inplace=True)
dataset['ADMIT TERM CODE'].fillna(dataset['ADMIT TERM CODE'].mode(), inplace=True)
dataset['ADMIT YEAR'] = dataset['ADMIT TERM CODE'].astype(str).str[:4]

dataset = dataset.groupby(["ID 2"]).apply(lambda x: x.sort_values(["ADMIT YEAR"], ascending=False)).drop_duplicates('ID 2',keep='first').reset_index(drop=True)


#dataset['CURRENT STAY STATUS'] = dataset['CURRENT STAY STATUS'].apply(change_STAY_STATUS)

print(dataset['CURRENT STAY STATUS'])

dataset.info()

print(dataset['PREV EDU CRED LEVEL NAME'].unique())
num_cols = ['PROGRAM SEMESTERS', 'TOTAL PROGRAM SEMESTERS', 'FIRST YEAR PERSISTENCE COUNT', 'ENGLISH TEST SCORE',
            'Term1', 'Term2', 'Term3','Term4', 'Term5', 'Term6','Term7', 'Term8', 'Term9', 'Term10']
cat_colls = ['INTAKE COLLEGE EXPERIENCE', 'SCHOOL CODE', 'STUDENT LEVEL NAME','MAILING CITY NAME', 'TIME STATUS NAME',
            'RESIDENCY STATUS NAME', 'FUNDING SOURCE NAME', 'GENDER', 'DISABILITY IND', 'MAILING COUNTRY NAME',
            'CURRENT STAY STATUS',  'ACADEMIC PERFORMANCE', 'AGE GROUP LONG NAME', 'APPLICANT CATEGORY NAME',
            'APPLICANT TARGET SEGMENT NAME', 'PREV EDU CRED LEVEL NAME','FIRST GENERATION IND']
cat_cols = ['INTAKE COLLEGE EXPERIENCE', 'SCHOOL CODE', 'STUDENT LEVEL NAME', 'TIME STATUS NAME',
            'RESIDENCY STATUS NAME', 'FUNDING SOURCE NAME', 'GENDER', 'DISABILITY IND', 'MAILING COUNTRY NAME',
            'CURRENT STAY STATUS',  'ACADEMIC PERFORMANCE', 'AGE GROUP LONG NAME', 'APPLICANT CATEGORY NAME',
            'APPLICANT TARGET SEGMENT NAME', 'PREV EDU CRED LEVEL NAME']
'''
for var in cat_cols:
    cat_list='var'+'_'+var
    print(cat_list)
    cat_list = pd.get_dummies(dataset[var],prefix=var,drop_first = False)
    #print(cat_list)
    df=dataset.join(cat_list)
    dataset=df
'''
dataset.shape

#dataset=dataset.drop(columns=cat_cols)

dataset.head()
#dataset = dataset.astype(float)
dataset.info()

X=dataset.drop(columns=['SUCCESS LEVEL'])

#X = np.asarray(X).astype(np.int32)
Y=dataset["SUCCESS LEVEL"]
print(Y)
Y.replace(('Successful', 'Unsuccessful'), (1, 0), inplace=True)

#Y=Y.astype(int)
#print(Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=59, stratify=Y)


x_axis, y_axis = y_train.unique()
g = sns.countplot(Y)

num_pipeline = Pipeline(steps=[
    ('numerical_transformer',NumericalTransformer(cols=num_cols)),
    ('impute',SimpleImputer(strategy='mean')),
    ('standard_scalar',StandardScaler())
])


cat_pipeline = Pipeline([
    ('cat_transformer',CategoricalTransformer(cols=cat_cols)),
    ('cat_impute',SimpleImputer(strategy='most_frequent')),
    ('one_hot_encoder',OneHotEncoder(sparse=False, handle_unknown = 'ignore'))
])

full_pipeline = FeatureUnion([
    ('numerical_pipeline',num_pipeline),
    ('categorical_pipeline',cat_pipeline)
])

pipe = Pipeline([
    ('full_pipeline',full_pipeline)
])

X_train = pipe.fit_transform(X_train)

X_test = pipe.transform(X_test)


print(f"X_train shape ={X_train.shape}")
print(f"X_val shape ={X_test.shape}")
print(f"y_train shape ={y_train.shape}")
print(f"y_val shape ={y_test.shape}")

"""
hype_model = tf.keras.Sequential([
  layers.Dense(32, activation='relu', input_shape=(108,), kernel_initializer='glorot_uniform'),
  layers.Dropout(rate=0.5),  
  layers.Dense(1, activation='sigmoid')
])

hype_model.compile(loss=tf.keras.losses.binary_crossentropy,metrics=['accuracy'],
                      optimizer = tf.optimizers.Adam())

history = hype_model.fit(X_train, y_train, epochs=100)



"""
n_epochs = 10
checkpoint_filepath = './'
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath + 'best.h5', 
    save_weights_only=True, monitor='val_accuracy', verbose = 1, mode='max', save_best_only=True)

model_early_stop = EarlyStopping(monitor='val_accuracy', verbose =1, mode='max', patience=2)

hype_model = tf.keras.Sequential([
  layers.Dense(32, activation='relu', input_shape=(70,), kernel_initializer='glorot_uniform'),
  layers.Dropout(rate=0.5),  
  layers.Dense(1, activation='sigmoid')
])

hype_model.compile(loss = tf.keras.losses.binary_crossentropy, metrics=['accuracy'],
                      optimizer = tf.optimizers.Adam())

hype_model.summary()

history = hype_model.fit(X_train, y_train, callbacks=[model_checkpoint_callback], epochs=n_epochs,validation_data=(X_test, y_test))

hype_model.predict(X_test)

print(y_test)


joblib.dump(history,"nn_group.pkl")

#save model
import pickle
filename = 'nn_group.pickle'
pickle.dump(history, open(filename, 'wb'))