
import pandas as pd
        
from sklearn.linear_model import LogisticRegression    

from sklearn.model_selection import train_test_split
import pickle

df=pd.read_csv('car.csv')
print(df.head())

x= df.iloc[:,:-1]
y=df.iloc[:,-1]
x_train,x_test,y_train,y_test= train_test_split(x,y, test_size = 0.2)

model = LogisticRegression()

model.fit(x_train,y_train)

# pred=model.predict([[72000,2]])
# print(pred)
pickle.dump(model,open('car.pkl','wb'))