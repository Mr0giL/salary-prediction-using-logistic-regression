import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
lable_encode = preprocessing.LabelEncoder()
ud_df = pd.read_excel('6th data.xlsx')
# print(ud_df.info())
y = ud_df['salary']
X = ud_df.drop('salary', axis= 1, inplace= False)
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=37)
x_test_d = pd.get_dummies(x_test)
x_train_d = pd.get_dummies(x_train)
y_test_d = lable_encode.fit_transform(y_test)
y_train_d = lable_encode.fit_transform(y_train)
print(y_test_d)
lgr = LogisticRegression(max_iter= 100000000000)
lgr.fit(x_train_d,y_train_d)
y_pred = lgr.predict(x_test_d)
test_score = accuracy_score(y_test_d,y_pred,normalize=True).astype(str)
f = open('prediction score.txt','w')
f.write('computing score : ')
f.write(test_score)
f.close()
ohe_to_category_dic = {0:'high', 1: 'low', 2: 'medium'}
y_pred_translated = [ohe_to_category_dic.get(i) for i in y_pred]
y_pred_translated = np.array(y_pred_translated).reshape(-1,1)
x_test['real sallary'] = y_test
x_test['predicted sallary'] = y_pred_translated
x_test.to_csv('results.csv')

