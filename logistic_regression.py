import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import sklearn

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('C:\Abraão\Educação\Huawei ICT Competition\Final LATAM\creditcard.csv') # Task 1

dataset = dataset.drop('Time', axis=1)
dataset = pd.DataFrame(dataset)

dataset.iloc[:, :5] 

data_for_plot = dataset['Class']

plt.hist(data_for_plot, bins=[0, 1, 2]) # Task 2

data_for_normalize = dataset['Amount']
data_for_normalize = np.array(data_for_normalize)

normalized = (data_for_normalize - data_for_normalize.mean()) / data_for_normalize.std() # Task 3

x = dataset.drop('Class', axis=1) # Elements
y = dataset['Class'] # Indexes

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, shuffle=True, test_size=0.15)

x_train.shape
x_test.shape
y_train.shape
y_test.shape

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

# RandomUnderSampler object
rus = RandomUnderSampler(random_state=42, sampling_strategy = 'majority')
x_undersample, y_undersample = rus.fit_resample(x_train, y_train) # Undesempled dataset

smote = SMOTE()
x_oversample, y_oversample = smote.fit_resample(x_train, y_train)

from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

lm = LogisticRegression(max_iter=2000)
lm_undersample = LogisticRegression(max_iter=2000)
lm_oversample = LogisticRegression(max_iter=2000)
folds = KFold(n_splits = 5, shuffle = True, random_state = 100)

lm.fit(x_train, y_train)
lm_undesample.fit(x_undersample, y_undersample)
lm_oversample.fit(x_oversample, y_oversample)

scores = cross_val_score(lm, x_train, y_train, scoring='r2', cv=folds, verbose=2)  
scores_undersample = cross_val_score(lm, x_undersample, y_undersample, scoring='r2', cv=folds, verbose=2)  
scores_oversample = cross_val_score(lm, x_oversample, y_oversample, scoring='r2', cv=folds, verbose=2)  

plt.plot(scores)
plt.plot(scores_undersample)
plt.plot(scores_oversample)
plt.title("Original, Undersample and Oversample comparasion")

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

y_pred = cross_val_predict(lm_oversample, x_oversample, y_oversample, cv=folds, verbose=1)
conf_mat = confusion_matrix(y_oversample, y_pred)
