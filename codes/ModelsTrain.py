import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
ogdata=pd.read_excel('ML.xlsx')
traindata=pd.read_excel('MLt.xlsx')

x=ogdata.iloc[:,:-1]
y=ogdata.iloc[:,-1]

model=RandomForestClassifier()
model.fit(x.values,y.values)

c=traindata.iloc[:,:-1].values
actual=traindata.iloc[:,-1]
model.predict(c)
actual.values
count=0
for i in range(len(actual)):
    if actual[i] == k[i]:
        count+=1
print((count*100)/len(actual))


train = pd.read_csv(r"C:\Users\kholi\Downloads\adult\adult.data",header=None)
test = pd.read_csv(r"C:\Users\kholi\Downloads\adult\adult.test",header=None)
x_train =train.iloc[:, :-1]
y_train =train.iloc[:, -1]
x_test =test.iloc[:, :-1]
y_test =test.iloc[:, -1]
encoder = OneHotEncoder(handle_unknown='ignore')
x_train_enc = encoder.fit_transform(X_train)
x_test_enc = encoder.transform(X_test)
model = RandomForestClassifier(random_state=42)
model.fit(X_train_enc, y_train)
y_pred = model.predict(X_test_enc)

