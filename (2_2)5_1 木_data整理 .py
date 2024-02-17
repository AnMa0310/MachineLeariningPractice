import pandas as pd 

df=pd.read_csv('iris.csv')
print(df.tail(3))#headは頭、tailは後ろから
print(df['種類'].unique()) #uniqueは種類列から重複取り除いた
print(df['種類'].value_counts())#.value_counts()で出現回数をカウント

#print(df.isnull()) 各マスが欠損値かどうか調べる
print(df.isnull().any(axis=0)) #axis=0は列ごと,axis=1は行ごとに欠損値があるかの確認
print(df.sum())

tmp=df.isnull()
print(tmp.sum())

print('CHANGE THE FUCK OVER')

df2=df.dropna(how='any',axis=0)#dropnaは欠損値があるdataを削除 anyは一つでも、allはすべて
print(df2.tail(3))
print(df.isnull().any(axis=0))

df['花弁長さ']=df['花弁長さ'].fillna(0)#fillnaで代入
print(df.tail(3))

print(df['花弁長さ'].mean())#欠損データは一般的に平均値を出し、穴埋めの部分に代入する
