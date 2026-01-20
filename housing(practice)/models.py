import pandas as pd 
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score # for avaoiding overfitting the data 
# read the data
df=pd.read_csv('housing.csv')

df['income_cat']=pd.cut(df['median_income'],bins=[0.0,1.5,3.0,4.5,6.0,np.inf],labels=[1,2,3,4,5])
# Split the data into 2 sets test and train 
split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=43)

for tr,tst in split.split(df,df['income_cat']):
    train=df.iloc[tr].drop('income_cat',axis=1)
    test=df.iloc[tst].drop('income_cat',axis=1)

data=train.copy()   

ylabel=data['median_house_value'].copy()
data=data.drop('median_house_value',axis=1)
# Sapereate the num_colmans anf cat_columns 
num_col=data.drop('ocean_proximity',axis=1).columns.tolist()
cat_col=['ocean_proximity']
# Things to do in the numrerical values
num_pipeline=Pipeline([
                     ('imputer',SimpleImputer(strategy='median')),
                     ('scaling',StandardScaler())
                ])
# things to do with the category values
cat_pipeline=Pipeline([
                 ('encoding',OneHotEncoder(handle_unknown='ignore'))
                       ])

# final pipeplines fit and transform 
full_pipeline=ColumnTransformer([('num',num_pipeline,  num_col),
                                 ('cat',cat_pipeline, cat_col)
                                 ]) 
final_data=full_pipeline.fit_transform(data)

# Model Fitting 
print('LinearRegression-')
linear=LinearRegression()
linear.fit(final_data,ylabel)
lin_preds=linear.predict(final_data)
linrms=-cross_val_score(linear,final_data,ylabel,scoring='neg_root_mean_squared_error',cv=5)
print(pd.Series(linrms).describe())
print()


print('DecisionTreeRegressor-')
tree=DecisionTreeRegressor()
tree.fit(final_data,ylabel)
line_preds=tree.predict(final_data)
tenrms=-cross_val_score(tree,final_data,ylabel,scoring='neg_root_mean_squared_error',cv=5)
print(pd.Series(tenrms).describe())
print()

print('RandomForestRegressor-')
randomm=RandomForestRegressor()
randomm.fit(final_data,ylabel)
line_=randomm.predict(final_data)
linrmss=-cross_val_score(randomm,final_data,ylabel,scoring='neg_root_mean_squared_error',cv=5)
print(pd.Series(linrmss).describe())

print('RandomForestRegressor')




