#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 15:23:46 2021

@author: fongnam
"""

from flask import Flask, make_response, request
import io
from io import StringIO
import csv
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV


app = Flask(__name__)

def transform(text_file_contents):
    return text_file_contents.replace("=", ",")


@app.route('/')
def form():
    return """
        <html>
            <body>
            <style>
                body{
                  background-image: url("")
                  background-size: auto;
                  background-position: center; 
                  background-size: 20%;
                  background-attachment: inherit;
                  background-repeat: no-repeat;
                  background-color: #E6E2E1;
                }
              </style>
            <div style="text-align: center;">
            <br>
            <h1 class="headertext">Customer Response Prediction</h1>
                </br>
                </br>
                <p> Insert your CSV file and then download the Result
                <form action="/transform" method="post" enctype="multipart/form-data">
            
                    <input type="file" name="data_file" class="btn btn-block"/>
                    </br>
                    <button type="submit" class="btn btn-primary btn-block btn-large">Submit</button>
                    </br>
                </form>
            </body>
        </html>
    """
@app.route('/transform', methods=["POST"])
def transform_view():
    f = request.files['data_file']
    if not f:
        return "No file"
    stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
    csv_input = csv.reader(stream)
    #print("file contents: ", file_contents)
    #print(type(file_contents))
    print(csv_input)
    for row in csv_input:
        print(row)

    stream.seek(0)
    result = transform(stream.read())

    df = pd.read_csv(StringIO(result), sep = ';')
    print(df)
    def impute_values(variable, values, weight):
        if variable == 'unknown':
            return np.random.choice(values, p=weight)
        else: 
            return variable 

    def fill_unknown(df, col):
        #for col in column:
        unique = list(set(df[col].values))
        unique.remove('unknown')
    
        weight = df[df[col] != 'unknown'][col].value_counts(normalize = True)
        weight = [i/sum(weight) for i in weight]
    
        df[col] = df[col].map(lambda x: impute_values(x, unique, weight))

    unknown_cols = ['job','marital', 'education', 'housing', 'loan']
    for col in unknown_cols:    
        try:
            fill_unknown(df,col)
        except ValueError:
            pass
    print('housing fill unknown',df['housing'].unique())
    nominal_yn = ['housing', 'loan']

    for col in nominal_yn:
        df[col] =df[col].map({'no':0, 'yes':1})
    print(df['housing'].unique())
        
    df['pdays'] = df['pdays'].map(lambda x: 0 if x == 999 else x)
    
    
    lst=['basic.9y','basic.6y','basic.4y']
    for i in lst:
        df.loc[df['education'] == i, 'education'] = "middle.school"
    
    
    df['education'] = df['education'].map(lambda x: 0 if x =='illiterate' else 1 if x =='middle.school' else 2 if x =='high.school' else 3 if x == 'university.degree' else 4)

 
    print(df.isnull().sum())
    
    df_dummy = pd.get_dummies(df[['job', 'contact', 'marital', 'poutcome', 'default', 'day_of_week', 'month']], drop_first = True)
    df_d = pd.concat([df.drop(columns = ['job', 'contact', 'marital', 'poutcome', 'default', 'day_of_week', 'month']), df_dummy], axis =1)
    drop_col = ['default_no','job_admin.','marital_divorced','poutcome_failure','contact_cellular', 'duration','housing', 'previous', 'loan', 'emp.var.rate']
    for col in drop_col:    
        try:
            df_d.drop(columns = [col], inplace = True)
        except KeyError:
            pass
    model_col_list = ['age', 'education', 'campaign', 'pdays', 'cons.price.idx',
                      'cons.conf.idx', 'euribor3m', 'nr.employed', 'job_blue-collar',
                      'job_entrepreneur', 'job_housemaid', 'job_management', 'job_retired',
                      'job_self-employed', 'job_services', 'job_student', 'job_technician',
                      'job_unemployed', 'contact_telephone', 'marital_married',
                      'marital_single', 'poutcome_nonexistent', 'poutcome_success',
                      'default_unknown', 'default_yes', 'day_of_week_mon', 'day_of_week_thu',
                      'day_of_week_tue', 'day_of_week_wed', 'month_aug', 'month_dec',
                      'month_jul', 'month_jun', 'month_mar', 'month_may', 'month_nov',
                      'month_oct', 'month_sep']
    for col in model_col_list:
        if col not in df_d.columns:
            df_d[col] = 0
    print('len df_d cols', df_d.columns) 
    ss = StandardScaler()
    ss.fit(pd.read_csv('x-train.csv'))
    df_d_sc = ss.transform(df_d)
    # load the model from disk
    with open('xg_model.pkl','rb') as h:
        loaded_model = pickle.load(h)
    #loaded_model = pickle.dump(lr_model,f)
    df['prediction'] = loaded_model.predict(df_d_sc)

    

    response = make_response(df.to_csv())
    response.headers["Content-Disposition"] = "attachment; filename=result.csv"
    return response

if __name__ == "__main__":
    app.run(debug=False,port=9000)