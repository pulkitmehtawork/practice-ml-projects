import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
df = pd.read_csv('BankNote_Authentication.csv')
print(df.head())
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)
rf = RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
score = accuracy_score(y_test,y_pred)
print(score)

pickle_out = open('classifier.pkl','wb')
pickle.dump(rf,pickle_out)
pickle_out.close()




