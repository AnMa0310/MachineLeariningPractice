import pandas as pd  
from sklearn import tree
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt    

"""決定木モデルは外れ値を外す行為はいらない。
しかし、重回帰モデルはデータから直線を引くモデルだから、外れ値を外す必要がある"""

"""不均衡データは処理が必須！！！
Survived
0    549
1    342

step1 欠損値の確認
step2 データの総数の確認
step3 データ欠損の穴埋め
step4 特徴量xと正解データtに分割
step5 訓練とテストデータに分割
step6 モデルの学習
step7 モデルのチューニング(一連の作業の繰り返しだから自分の(learn)関数を作る！)

＊モデルチューニングの注意点
過学習：モデルを必要以上に複雑にすると、テストデータの正解率が下がる。
理由は訓練データのとても細かい(必要以上の)特徴までも勉強する。

過学習の解決策:1.データの数と読み取るデータの数と正確性を増やす
 1)
2.データの前処理の仕方を変える(1平均値ではなく、中央値を入れる。小グループ単位で見る。)
3.モデルの学習時も設定を変える
4.そもそもの分析手法を変える"""


df=pd.read_csv('Survived.csv')
print(df.head(2))

print(df['Survived'].value_counts())

print(df.isnull().sum())#欠損値の確認

print(df.shape)#(891行,11列)

df['Age']=df['Age'].fillna(df['Age'].mean())#Age列を平均値で穴埋め
print(df['Embarked'].unique())
print(df['Embarked'].value_counts())
#文字列のstr型は平均値を取るやり方ではなく、df['Embarked']=df['Embarked'].fillna(df['Embarked'].mean())
print(df['Embarked'].mode()[0])#mode()[0]でensures that you get the most frequent value.
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

col=['Pclass','Age','SibSp','Parch','Fare']
x=df[col]
t=df['Survived']

xtrain,xtest,ytrain,ytest=train_test_split(x,t,test_size=0.2,random_state=0)
print(xtrain.shape)

model=tree.DecisionTreeClassifier(max_depth=5,random_state=0,class_weight='balanced')
#class_weight='balanced'を設定することで不均衡の比率の小さいデータの影響を大きくする。
model.fit(xtrain,ytrain)

print(model.score(xtest,ytest))

def learn(x,t,depth=3):#もしdepthが指定されていなかったら3にしてくれる。
    xtrain,xtest,ytrain,ytest=train_test_split(x,t,test_size=0.2,random_state=0)
    model=tree.DecisionTreeClassifier(max_depth=depth,random_state=0,class_weight='balanced')
    model.fit(xtrain,ytrain)

    score=model.score(xtrain,ytrain)
    score2=model.score(xtest,ytest)
    return round(score,3),round(score2,3),model #round関数は小数点3桁を四捨五入する

for j in range(1,15):#range(start, stop[, step])
    trainscore,testscore,model=learn(x,t,depth=j)
    sentence='訓練データの正解率{}'
    sentence2='テストデータの正解率{}'
    totalsentence='深さ{}: '+sentence+sentence2 
    print(totalsentence.format(j,trainscore,testscore))

#データの前処理の仕方を変える(1平均値ではなく、中央値を入れる。小グループ単位で見る。

df2=pd.read_csv('Survived.csv')

df2['Age'] = df2['Age'].fillna(df2['Age'].mean())
print(df2.dtypes)

#print(df2['Age'].unique())#uniqueとは一意で重複のないという意味

n=['Survived','Age']
df3=df2[n]


print(df3.groupby('Survived').mean())
#小グループの基準となる列の作成:変数.groupby('列名')
#*groupbyの注意点として変数にオブジェクト型が一つでもあるとエラー出る。

m=['Pclass','Age']
df4=df2[m]

print(df4.groupby('Pclass').mean()['Age'])

df3=pd.read_csv('Survived.csv')

print(pd.pivot_table(df3,index='Survived',columns='Pclass',values='Age'))
#pivot_tableで基準となる二つの列を使った集計を、行いvalueのデフォルトで平均値を出す
print(pd.pivot_table(df3,index='Survived',columns='Pclass',values='Age',aggfunc=max))#リストの最大値を出す。

is_null=df3['Age'].isnull()#欠損値を,0False,1False,2True...のようにしてくれる。


df3.loc[(df3['Pclass']==1)&(df3['Survived']==0)&(is_null),'Age']=43.6
df3.loc[(df3['Pclass']==2)&(df3['Survived']==0)&(is_null),'Age']=33.5
df3.loc[(df3['Pclass']==3)&(df3['Survived']==0)&(is_null),'Age']=26.5

df3.loc[(df3['Pclass']==1)&(df3['Survived']==1)&(is_null),'Age']=35.3
df3.loc[(df3['Pclass']==2)&(df3['Survived']==1)&(is_null),'Age']=25.9
df3.loc[(df3['Pclass']==3)&(df3['Survived']==1)&(is_null),'Age']=20.6

col=['Pclass','Age','SibSp','Parch','Fare']
x1=df3[col]
t1=df3['Survived']

for n in range(1,15):
    s1,s2,m=learn(x1,t1,depth=n)
    sentence='深さ{}:訓練データの精度{}::テストデータの精度{}'
    print(sentence.format(n,s1,s2))


col1=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
col2=['Survived']
nf=df3[col1]
tf=df3['Survived']


male=pd.get_dummies(nf['Sex'],drop_first=True,dtype=int)#get_dummiesで文字列を指定する。
print(male)
Embark=pd.get_dummies(nf['Embarked'],drop_first=True,dtype=int)
print(Embark)

x_temp=pd.concat([nf,male,Embark],axis=1)#データフレームを横方向に連結
print(x_temp)

x_new=x_temp.drop(['Embarked','Sex'],axis=1)


print(x_new)

for j in range(1,15):
    s1,s2,m=learn(x_new,tf,depth=j)
    s='深さ{}:訓練データの精度{}::テストデータの精度{}'
    print(s.format(j,s1,s2))

s1,s2,model=learn(x_new,tf,depth=8)

print(pd.DataFrame(model.feature_importances_,index=x_new.columns))
#feature_importancesでモデルの特徴量重要度を変換する。


plot_tree(model,feature_names=x_new.columns,filled=True)#Trueは大文字じゃないと箱と認識する
plt.show()


#train_score,test_score,model=learn(nf,tf)

#print(nf.dtypes)
#print(nf['Sex'].value_counts())

#sex = nf.groupby('Sex').mean()
#print(sex['Survived'])

#sex['Survived'].plot(kind='bar')
#plt.show()
"""n=['Sex','Survived']
sex=df3[n]
print(sex.groupby('Survived').mean())"""
