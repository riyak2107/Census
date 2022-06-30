#Census prediction

import pandas as pd

#from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from  sklearn.metrics import accuracy_score
#from sklearn.metrics import classification_report
#from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
#from sklearn import svm
# from sklearn.neural_network import MLPClassifier
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.naive_bayes import GaussianNB

#lr=LogisticRegression(random_state=0,solver='sag',max_iter=10000)
rf=RandomForestClassifier(random_state=1)
# gbm=GradientBoostingClassifier(n_estimators=10)
dt=DecisionTreeClassifier(random_state=0)
# sv=svm.SVC()
# nn=MLPClassifier(solver='lbfgs', alpha=1e-4, hidden_layer_sizes=(1,1), random_state=0)
# nb=MultinomialNB()
# gb=GaussianNB()

df=pd.read_csv("C:/Users/Riya/Downloads/adult.csv")
#print(df.describe())
df['income'] = df['income'].replace({'<=50K': 0, '>50K':1}, regex=True)
#print(df)
df = pd.get_dummies(df)
x=df.drop("income",axis=1)
y=df["income"]
#print(df)

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.3)

#print(x_train)
#print(x_test)
#print(y_train)
#print(y_test)


#lr.fit(x_train,y_train)
#y_lrp=lr.predict(x_test)

rf.fit(x_train,y_train)
y_rfp=rf.predict(x_test)
#
#gbm.fit(x_train,y_train)
#y_gbmp=gbm.predict(x_test)
#
dt.fit(x_train,y_train)
y_dtp=dt.predict(x_test)
#
# sv.fit(x_train,y_train)
# y_svp=sv.predict(x_test)
#
# nn.fit(x_train,y_train)
# y_nnp=nn.predict(x_test)
#
# nb.fit(x_train,y_train)
# y_nbp=nb.predict(x_test)
#
# gb.fit(x_train,y_train)
# y_gbp=gb.predict(x_test)

#print('Logistic Regression : ' ,accuracy_score(y_test,y_lrp))
print('Random forest : ' ,accuracy_score(y_test,y_rfp))
# print('Gradient Boosting Method : ' ,accuracy_score(y_test,y_gbmp))
print('Decision Tree : ' ,accuracy_score(y_test,y_dtp))
# print('SVM : ' ,accuracy_score(y_test,y_svp))
# print('Neural Networks : ' ,accuracy_score(y_test,y_nnp))
# print('Naive Bayes : ' ,accuracy_score(y_test,y_nbp))
# print('GaussianNB : ' ,accuracy_score(y_test,y_gbp))

#Feature engineering using over sampling

from imblearn.over_sampling import RandomOverSampler
ros=RandomOverSampler(random_state=0)
x,y=ros.fit_resample(x,y)

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.3)

#print(x_train)
#print(x_test)
#print(y_train)
#print(y_test)


#lr.fit(x_train,y_train)
#y_lrp=lr.predict(x_test)

rf.fit(x_train,y_train)
y_rfp=rf.predict(x_test)
#
# gbm.fit(x_train,y_train)
# y_gbmp=gbm.predict(x_test)
#
dt.fit(x_train,y_train)
y_dtp=dt.predict(x_test)
#
# sv.fit(x_train,y_train)
# y_svp=sv.predict(x_test)
#
# nn.fit(x_train,y_train)
# y_nnp=nn.predict(x_test)
#
# nb.fit(x_train,y_train)
# y_nbp=nb.predict(x_test)
#
# gb.fit(x_train,y_train)
# y_gbp=gb.predict(x_test)

print('After feature engineering')
#print('Logistic Regression : ' ,accuracy_score(y_test,y_lrp))
print('Random forest : ' ,accuracy_score(y_test,y_rfp))
# print('Gradient Boosting Method : ' ,accuracy_score(y_test,y_gbmp))
print('Decision Tree : ' ,accuracy_score(y_test,y_dtp))
# print('SVM : ' ,accuracy_score(y_test,y_svp))
# print('Neural Networks : ' ,accuracy_score(y_test,y_nnp))
# print('Naive Bayes : ' ,accuracy_score(y_test,y_nbp))
# print('GaussianNB : ' ,accuracy_score(y_test,y_gbp))

'''
OUTPUT :
Random forest :  0.8514689323369843
Decision Tree :  0.8192240761592794
After feature engineering
Random forest :  0.9192961165048543
Decision Tree :  0.9103964401294499

Process finished with exit code 0
'''