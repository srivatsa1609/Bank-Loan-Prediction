
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df1=pd.read_csv('UniversalBank.csv')

print(df1.iloc[:7,:5])

df1.drop(['ID','ZIP Code'],axis=1,inplace=True)

print(df1.iloc[:7,:5])

print((df1['Age']).dtype)

df1['Age']=pd.to_numeric(df1['Age'])

print((df1['Age']).dtype)

sns.set_style('whitegrid')
sns.distplot(df1['Age'],rug=False,kde=True,color='tomato',label='Normally distributed age parameter')
plt.title('Distribution of age')
plt.legend(loc='Upper Left')
plt.show()
 
sns.set_style('darkgrid')
pal=sns.color_palette('PuOr')
sns.catplot(x='Family',hue='Personal Loan',kind='count',data=df1,palette=pal)
plt.title('Personal Loan based on different family sizes')
plt.show()


sns.set_style('ticks')
sns.scatterplot(x='Income',y='CCAvg',data=df1,size='Family',hue='Personal Loan',palette='tab20c_r')
plt.title('Income Vs CCAvg')
plt.show()


sns.set_style('dark')
sns.catplot(x='Personal Loan',y='CCAvg',data=df1,kind='bar',col='Securities Account',row='CD Account',palette='twilight_shifted_r')
plt.show()

sns.set_style('darkgrid')
sns.catplot(x='Education',hue='Personal Loan',kind='count',data=df1,palette='Wistia')
plt.title('Personal Loan based on Education level')
plt.show()

import numpy as np
import sklearn 
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

x=df1.drop(['Personal Loan'],axis=1).copy()
y=df1['Personal Loan'].values.reshape(-1,1)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=44)

lreg=LogisticRegression(solver='lbfgs')

lreg.fit(x_train,y_train.ravel())

y_predict=lreg.predict(x_test)

print('The first few predicted values (0 or 1) :',y_predict[:10])

y_prob=lreg.predict_proba(x_test)[:,1]
y_prob.ravel()
print('the first few actual probable values of the test data',y_prob[:10])

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

score=accuracy_score(y_test,y_predict)
print('The accuracy of the model is :',score)

matrix=confusion_matrix(y_test,y_predict)
print('The primary confusion matrix is : ',matrix)

from sklearn.metrics import classification_report 
from sklearn.metrics import roc_curve,auc

print(classification_report(y_test,y_predict))

fpr,tpr,thresholds=roc_curve(y_test,y_prob)

roc_auc=auc(fpr,tpr)
print('auc score :',roc_auc)

plt.plot(fpr,tpr,label='ROC-curve with accuracy score : %.2f' % roc_auc)
plt.title('ROC-curve')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(loc='lower right')
plt.show()

i=np.arange(len(tpr))
threshold_optimal=pd.DataFrame({'tf':pd.Series(tpr-(1-fpr),index=i),
'thresholds':pd.Series(thresholds,index=i),'gmean':pd.Series(np.sqrt(tpr*(1-fpr)),index=i)})
print(threshold_optimal.head())

threshold_optimal_1=threshold_optimal.iloc[(threshold_optimal.tf-0).abs().argsort()[:1]]
print('The 1st optimized threshold value :',threshold_optimal_1)

threshold_optimal_2=threshold_optimal.iloc[threshold_optimal['gmean'].argmax()]
print('The 2st optimized threshold value using G-mean :',threshold_optimal_2)

from sklearn.preprocessing import binarize

y_optimized=binarize(y_prob.reshape(1,-1),0.070244)[0]

matrix_1=confusion_matrix(y_test,y_optimized)
print(matrix_1)

print(classification_report(y_test,y_optimized))
