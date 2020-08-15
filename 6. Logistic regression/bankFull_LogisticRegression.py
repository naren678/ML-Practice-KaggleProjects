import pandas as pd
import seaborn as sb

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt 
le=LabelEncoder() 

bank = pd.read_csv("C:\\Users\\User\\Desktop\\assignments\\bank-full.csv")

print(len(bank))
bank.info()
bank.head()
bank.groupby('y').size()
bank['education'].unique()
bank['y'].value_counts()
sb.countplot(x='y',data=bank,palette='his')
bank.isnull().sum()
bank.shape()
bank['housing']=le.fit_transform(bank['housing'])
bank['default']=le.fit_transform(bank['default'])
bank['loan']=le.fit_transform(bank['loan'])
bank['marital']=le.fit_transform(bank['marital'])
bank['y']=le.fit_transform(bank['y'])
bank.drop(["month"],inplace=True,axis = 1)
bank.drop(["poutcome"],inplace=True,axis = 1)
bank.drop(["previous"],inplace=True,axis = 1)
bank.drop(["contact"],inplace=True,axis = 1)

bank.drop(["pdays"],inplace=True,axis = 1)
bank.drop(["default"],inplace=True,axis = 1)
bank.drop(["education"],inplace=True,axis = 1)
bank.drop(["job"],inplace=True,axis = 1)
bank.drop(["duration"],inplace=True,axis = 1)
bank.drop(["marital"],inplace=True,axis = 1)
bank.education.unique()

y=bank.iloc[:,-1]
x=bank.iloc[:,1:-1]
x.head()

model = ExtraTreesRegressor()
model.fit(x,y)
print(model.feature_importances_)

feat_importances = pd.Series(model.feature_importances_, index=x.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()

bank.hist(bins=10, figsize=(14,10), color='#E14906')
plt.show()

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.coef
classifier.fit(x,y)
y_pred = classifier.predict(x)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y,y_pred)
print (confusion_matrix)
##a=(39917+2)/(39917+5+2+5287)
