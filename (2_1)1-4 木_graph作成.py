import pandas as pd 
from sklearn import tree
from sklearn.tree import plot_tree
import pickle
import matplotlib.pyplot as plt


data={'Matsuda work time':[160,160],
      'ando work time':[40,40]}

df=pd.DataFrame(data) #dataframeは列と行を示す
df.index=['April','June'] #indexの変更

#print(df)
#print(df.shape)

df2=pd.DataFrame(data,index=['fuck','u'],columns=['Matsuda work time','ando work time'])
#変数名=pd.DataFrame(二次元のデータ、index=インデックスのリスト、columns=列名のリスト)
#print(df2)

df3=pd.read_csv('Kvst.csv')
#print(type(df3))
#print(df3.head(3)) #headの3は上からの3行
#print(df3['身長'].head(3)) #[]は列名
#print(df3['派閥'].head(3))

xcol=['身長','体重','年代']
x=df3[xcol]
#print(x)

t=df3['派閥']
#print(t)

model= tree.DecisionTreeClassifier(random_state=0)
model.fit(x,t)

taro=[170,70,20]
masa=[158,42,20]

new=[taro,masa]
#print(model.predict(taro))
print(model.predict(new))
print(model.score(x,t))

x.columns=['height','weight','ages']
plot_tree(model,feature_names=x.columns,filled=True)#Trueは大文字じゃないと箱と認識する
plt.show() #これをしないと出ないーよーーーーん

import pickle
with open('kinokoTakenoko.pkl','wb')as f:
    pickle.dump(model,f)