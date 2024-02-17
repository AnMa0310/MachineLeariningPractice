import pandas as pd 
import numpy as np  
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
import pickle


df=pd.read_csv('iris.csv')

xcol=['がく片長さ','がく片幅','花弁長さ','花弁幅']
df1=df[xcol]
t=df['種類']

colmean=df1.mean()

df3=df1.fillna(colmean) #df.fillna(穴埋めの値)
print(df3.isnull().any(axis=0))

print(df3)

model=tree.DecisionTreeClassifier(max_depth=5,random_state=0)#max_depthはtreeの深さ
model.fit(df3,t)#モデルの学習
print(model.score(df3,t))

"""この時点での学習はダメで
ホールドアウト：訓練データとテストデータが必要。
これはfrom sklearn.model_selection import train_test_split"""

xtrain,xtest,ytrain,ytest=train_test_split(df3,t,test_size=0.3,random_state=0)
#test sizeは割り合い,random_stateは乱数シードでこれはコンピューター上のもの基本model定義の部分と一緒にする。
print(xtrain.shape)
print(xtest.shape)

model.fit(xtrain,ytrain)
"""modelは常々上書き保存される"""
print(model.score(xtest,ytest))

with open('irismodel.pkl','wb') as f:
    pickle.dump(model,f)
print('切り替え')
print(model.tree_.feature) 

"""[ 3 -2  3 -2 -2] 3は条件に使った列,-2は条件終了.
[ノードの割り当て]
0が一つ目の葉と同様
[列の割り当て]
0はがく片長さ 1はがく片幅 2は花弁長さ 3は花弁幅"""

print(model.tree_.threshold)
print('切り替え')

print(model.classes_)
print(model.tree_.value[1].astype(int))#ここの表示が謎、本来なら34
print(model.tree_.value[3].astype(int))
print(model.tree_.value[4].astype(int)) 


"""このツリーを描画することができるが、英語に変換する"""

xtrain.columns=['gaku_nagasa','gaku_haba','kaben_nagasa','kaben_haba']
plot_tree(model,feature_names=xtrain.columns,filled=True)#Trueは大文字じゃないと箱と認識する
plt.show() #これをしないと出ないーよーーーーん

with open('irismodel.pkl','wb') as f:
    pickle.dump(model,f)
print('切り替え')

morichan=np.array([0.32,0.8,0.56,0.30])
morichan_reshape=morichan.reshape(1,-1)
print(model.predict(morichan_reshape))