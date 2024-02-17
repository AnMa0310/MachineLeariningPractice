import pandas as pd  
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

"""
エラーはシンプルなミスが多い！！！()リスト入れる等
1.訓練データ：学習に使用するデータ
2.検証データ：チューニングの参考にするデータ
3.最終的な学習済みモデルの予測性能を評価するためのデータ"""

df=pd.read_csv('Boston.csv')
print(df['CRIME'].value_counts())
crime=pd.get_dummies(df['CRIME'],drop_first=True,dtype=int)
print(crime)

df2=pd.concat([df,crime],axis=1)#axis=1は列方向
df2=df2.drop(['CRIME'],axis=1)

train_val,test=train_test_split(df2,test_size=0.2,random_state=0)

train_val_mean=train_val.mean()
train_val2=train_val.fillna(train_val_mean)
print(train_val2.isnull().sum())

colname=train_val2.columns
for name in colname:
    train_val2.plot(kind='scatter',x=name,y='PRICE')

#外れ値の確認
out_line1=train_val2[(train_val2['RM']<6)&(train_val2['PRICE']>40)].index
out_line2=train_val2[(train_val2['PTRATIO']>18)&(train_val2['PRICE']>40)].index
print(out_line1,out_line2)

train_val3=train_val2.drop([76],axis=0)
col=['INDUS','NOX','RM','PTRATIO','LSTAT','PRICE']

train_val4=train_val3[col]
print(train_val4.head(3))

"""列同士の相関係数を調べる。正の相関:片方の値が増加すると、もう一方も増加する傾向がある。
corrだと、相関の値が+-で表示される。
corrとabsで相関関係を出す際には、必ず外れ値を外す必要がある。
また各データは標準化を行う必要がある。from sklearn.preprocessing import StandardScaler
＊注意事項：検証用データは検証用データの平均値と標準偏差を使用するのはダメ！
理由：検証用データの平均値と標準偏差を使用することは、モデルが実際の予測を行う際には利用できない情報を使用していることになる。
"""
se=train_val4.corr()
print(se.map(abs))

train_cor=train_val4.corr()['PRICE']
print(train_cor)

abs_cor=train_cor.map(abs)
print(abs_cor.sort_values(ascending=False))#ascendingはTrueは昇順、Falseは降順

col=['RM','LSTAT','PTRATIO']
x=train_val4[col]
t=train_val4['PRICE']

x = pd.DataFrame(x)
t = pd.DataFrame(t)

xtrain,xval,ytrain,yval=train_test_split(x,t,test_size=0.2,random_state=0)

sc_model_x=StandardScaler()
sc_model_x.fit(xtrain)#各特徴量の平均値と標準偏差が計算されます。

sc_model_y=StandardScaler()
sc_model_y.fit(ytrain)

sc_x=sc_model_x.transform(xtrain)#各特徴量から平均を引き、標準偏差で割ることで、特徴量が標準化。
tmp_df=pd.DataFrame(sc_x,columns=xtrain.columns)
print(tmp_df.std())

sc_y=sc_model_y.transform(ytrain)#各特徴量から平均を引き、標準偏差で割ることで、特徴量が標準化。
tmp_df=pd.DataFrame(sc_x,columns=xtrain.columns)

model=LinearRegression()
model.fit(sc_x,sc_y)

def learn(x,t):
    #データの標準化
    xtrain,xval,ytrain,yval=train_test_split(x,t,test_size=0.2,random_state=0)
    sc_model_x=StandardScaler()
    sc_model_y=StandardScaler()
    sc_model_x.fit(xtrain)
    sc_x_train=sc_model_x.transform(xtrain)
    sc_model_y.fit(ytrain)
    sc_y_train=sc_model_y.transform(ytrain)
    #学習
    model=LinearRegression()
    model.fit(sc_x_train,sc_y_train)
    #検証データの標準化
    sc_x_val=sc_model_x.transform(xval)
    sc_y_val=sc_model_y.transform(yval)
    #訓練データと検証データの決定係数計算
    trainscore=model.score(sc_x_train,sc_y_train)
    valscore=model.score(sc_x_val,sc_y_val)

    return trainscore,valscore 

x=train_val3.loc[:,['RM','LSTAT','PTRATIO']]
t=train_val3[['PRICE']]

s1,s2=learn(x,t)
print(s1,s2)



#例)map関数
def celsius_to_fahrenheit(celsius):
    return (celsius * 9/5) + 32
temperatures_celsius = [0, 10, 20, 30, 40]
# Use map() to apply the conversion function to each temperature in the list
temperatures_fahrenheit = map(celsius_to_fahrenheit, temperatures_celsius)
# Convert the map object to a list because it will be unreadable way of text(identify値の表示になる)
temperatures_fahrenheit = list(temperatures_fahrenheit)
print(temperatures_fahrenheit)

#plt.show()