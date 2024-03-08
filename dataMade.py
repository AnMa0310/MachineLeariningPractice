import pandas as pd 
import matplotlib.pyplot as plt    

df=pd.read_csv('cinema.csv')
print(df.isnull().any(axis=0))#欠損値の値を算出

df2=df.fillna(df.mean())
print(df2.isnull().any(axis=0))

"""df2.plot(kind='scatter',x='SNS1',y='sales')
df2.plot(kind='scatter',x='SNS2',y='sales')
df2.plot(kind='scatter',x='actor',y='sales')
df2.plot(kind='scatter',x='original',y='sales')"""

no=df2[(df2['SNS2']>1000)& (df2['sales']<8500)].index
df3=df2.drop(no, axis=0) #noの条件でdropで行(axis=0)を取り除いている
#df.dropna(how='any',axis=0)#dropnaは欠損値があるdataを削除 anyは一つでも、allはすべて
print(df3.shape)

for name in df.columns:
    if name=='cinema_id'or name=='sales':
        continue #continueはskipのような意味。ここではcinema_idかsalesはスキップ
    df3.plot(kind='scatter',x=name,y='sales')

print('外れ値の削除の仕方')

plt.show()

test=pd.DataFrame({'Acolumn':[1,2,3],'Bcolumn':[4,5,6]})
print(test) #行を表示
print(test['Acolumn']<2) #条件付きで行のみを表示する
print(test[test['Acolumn']<2])#testを表示するが行を抜き出す。
print('切り替え')
print(test.drop(0,axis=0))
print(test.drop('Bcolumn',axis=1))#axis=1は列を示す

print(df[(df['SNS2']>1000)&(df['sales']<8500)].index)
#indexとはIndex([30], dtype='int64')1450,0,0,8626.162638,0,9229

col=['SNS1','SNS2','actor','original']
x=df3[col]

t=df3['sales']
print(df3.loc[2,'SNS1'])

index=[2,4,6]
col1=['SNS1','actor']
print(df3.loc[index,col1])#locは特定の行、列を
print(df3.loc[0:3,col1])#0:3は0以上3未満を出してくれる
#print(df3.loc[:,col1])は:hが全て