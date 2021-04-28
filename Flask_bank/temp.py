import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
import pickle

bank = pd.read_csv('bank_train_dummied.csv')

#bank_dummy = pd.get_dummies(bank[['job', 'contact', 'marital', 'poutcome', 'default']], drop_first = True)
#bank_d = pd.concat([bank.drop(columns = ['job', 'contact', 'marital', 'poutcome', 'default']), bank_dummy], axis =1)


#Call the Get_dummies function
X = bank.drop(columns = ['target','duration','housing', 'previous', 'loan', 'emp.var.rate'])
print(len(X.columns))
print(len(X.columns))
y = bank['target']
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.1, random_state=42, stratify = y) 

ss = StandardScaler()
X_train_sc = ss.fit_transform(X_train)
X_test_sc = ss.transform(X_test)
pipe_xg = Pipeline([('sm', SMOTE(random_state = 42)),('xg', XGBClassifier(random_state = 42))])
pipe_xg_params = {
    'xg__learning_rate': [0.01, 0.1, 0.2],
    'xg__max_depth': [3,5],
    'xg__min_child_weight': [2,4],
    'sm__k_neighbors': [3,5]}
xg = GridSearchCV(pipe_xg,param_grid = pipe_xg_params,cv=5,n_jobs= -1,scoring='roc_auc',verbose=10)
xg.fit(X_train_sc, y_train)

# save the model to disk
with open('xg_model.pkl','wb') as f:
  pickle.dump(xg,f)
#pickle.dump(lr, open('lr_model.pkl', 'wb'))

