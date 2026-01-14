import pandas as pd 
data=pd.read_csv("housing.csv")
import matplotlib.pyplot as plt 
len(data)
# def test(data,name,ratio):
#     test_rows= int(ratio*len(data))
#     test_data=data.sample(test_rows,random_state=32) 
#     train_data=data.drop(test_data.index)
#     test_csv=test_data.to_csv(f"{name}test.csv",index=False)
#     train_csv=train_data.to_csv(f"{name}train.csv",index=False)
#     return test_data
# train=pd.read_csv('housingtrain.csv')
# test=pd.read_csv('housingtest.csv')
# train.head()
import numpy as np
# data['median_cate']=pd.cut(data['median_income'],bins=[0,1.3,3.0,4.5,6.0,np.inf],labels=[1,2,3,4,5]) # no new columns if a clumn is alreay categorical
from sklearn.model_selection import StratifiedShuffleSplit as Sp
modell=Sp(n_splits=1,test_size=0.2,random_state=42)
for train , test in modell.split(data,data['median_cate']) :
    train_set=data.loc[train]
    test_set=data.loc[test]
train_set
test_set
# for col in (test_set,train_set):
#     col.drop('median_cate',axis=1,inplace=True)
train_sett=train_set.copy()
train_sett.plot(kind='scatter',x='latitude',y='longitude',alpha=0.2)
from sklearn.impute import SimpleImputer as si
imputer=si(strategy='median')
housing_num= train_set.select_dtypes(include=[np.number])
imputer.fit(housing_num)
housing_num[housing_num['total_bedrooms'].isna()]
housing_num.info()
x=imputer.transform(housing_num)
