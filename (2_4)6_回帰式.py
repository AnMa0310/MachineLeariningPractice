#特徴量が1個だけ→線形単回帰分析 特徴量が2個以上→線形重回帰分析
#ex) 100*(SNS1)+20*(SNS2)+7*(actor)+10*(original)+1000
#回帰分析では最小2乗法で、直線を求める。

#MSE:差分を二乗したデータでの平均値
#MAE:差分の絶対値の平均値 from sklearn.metrics import mean_absolute_error

import pandas as pd 
import matplotlib.pyplot as plt    
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


df=pd.read_csv('cinema.csv')
df2=df.fillna(df.mean())
no=df2[(df2['SNS2']>1000)& (df2['sales']<8500)].index
df3=df2.drop(no, axis=0) 

col=['SNS1','SNS2','actor','original']
x=df3[col]
t=df3['sales']

model=LinearRegression() #maxもrandomもいらない
x_train,x_test,y_train,y_test=train_test_split(x,t,test_size=0.2,random_state=0)

model.fit(x_train,y_train)
print(model.score(x_test,y_test))
#score()は値が予測値と実測値の誤差。大きいほど誤差が少なく、0.8以上であれば良いモデル。

new=[[150,700,300,0]]
print(model.predict(new))

pred=model.predict(x_test)
print(mean_absolute_error(y_pred=pred,y_true=y_test))
#mean_absolute_error(y_pred=予測結果のデータ,y_true=実際のデータ))
#予測と実際の値の誤差が27万円ということ

print(model.coef_)#計算式の係数の表示
print(model.intercept_)#計算式の切片の表示

tmp=pd.DataFrame(model.coef_)
tmp.index=x_train.columns
print(tmp)

with open('cinema.pkl','wb') as f:
    pickle.dump(model,f)