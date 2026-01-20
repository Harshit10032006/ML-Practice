import os 
import numpy as np
import pandas as pd 
import joblib 
from sklearn.model_selection import StratifiedShuffleSplit #for train and test set
from sklearn.preprocessing import OneHotEncoder # for encoding cat into num
from sklearn.preprocessing import StandardScaler # for scaling the data for precise output 
from sklearn.impute import SimpleImputer # Nan vlues
from sklearn.pipeline import Pipeline # pipeline fast 
from sklearn.compose import ColumnTransformer # operations on selected columns only 
# from sklearn.linear_model import LinearRegression 
# from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor # Ml MOdel for used for prediction 
# from sklearn.metrics import root_mean_squared_error  # for checking the accuracy of the model 
# from sklearn.model_selection import cross_val_score # for avaoiding overfitting the data 

MODEL_FILE='model.pkl'
PIPELINE_FILE='pipeline.pkl'

def build_pipeline(num_col,cat_col):
    num_pipeline = Pipeline([('imputer',SimpleImputer(strategy='median')),
                             ('scaliing',StandardScaler())
                             ])
    cat_pipeline=Pipeline([('Encoding',OneHotEncoder(handle_unknown='ignore'))
                           ])


    full_pipeline= ColumnTransformer([('num',num_pipeline,num_col),
                                      ('cat',cat_pipeline,cat_col)
                                      ])
    

    return full_pipeline


if not os.path.exists(MODEL_FILE) :
    housing=pd.read_csv(r'C:\Users\kholi\datascience\numpy\Machine learning\housing.csv')
    housing['income_cat']=pd.cut(housing['median_income'],bins=[0.0,1.5,3.0,4.5,6.0,np.inf],labels=[1,2,3,4,5])
    split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
    for trn,tst in split.split(housing,housing['income_cat']) :
        housingg= housing.loc[trn].drop('income_cat',axis=1)
        housing.loc[tst].drop('income_cat',axis=1).to_csv('input.csv',index=False)

    ylabel=housingg['median_house_value'].copy()
    housingg=housingg.drop('median_house_value',axis=1)

    cat_col=['ocean_proximity']
    num_col=housingg.drop('ocean_proximity',axis=1).columns.tolist()

    pipeline=build_pipeline(num_col,cat_col)
    housing_fin=pipeline.fit_transform(housingg)


    model=RandomForestRegressor(random_state=43)
    model.fit(housing_fin,ylabel)

    joblib.dump(model,MODEL_FILE)
    joblib.dump(pipeline,PIPELINE_FILE)

    print("Done")

else :
    model=joblib.load(MODEL_FILE)
    pipeline=joblib.load(PIPELINE_FILE)

    input_data=pd.read_csv('inputcopy.csv')
    transformed_input=pipeline.transform(input_data) # pipeline alreday know waht to do 
    predictions=model.predict(transformed_input) # model is already trained 
    input_data['median_house_value']=predictions # saves the predicted values in the new column
    input_data.to_csv('ouput.csv',index=False)
    print("Done")

