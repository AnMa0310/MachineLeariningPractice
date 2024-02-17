import tkinter as tk 

win=tk.Tk()#ウインドウの作成
win.title('LETS DO THIS SHIT')#ウインドウのタイトル
win.minsize(340,240)#ウインドウのサイズ

label=tk.Label(win,text='青い空の下',bg='white',fg='red')#ラベルの作成。bgはラベルの色,fgは文字の色
label.place(x=100,y=100)#ラベルの配置。左上が(0,0),xが正方向,yが下向きの方法

def push():
    print('fuk')
    c.create_rectangle(10,10,60,60,fill='white')

btn=tk.Button(win,text='u wanna have fuk',command=push)
#第1引数は配置するウインドウ、第2引数はボタンに表示するウィンドウ
btn.place(x=0,y=0)

c = tk.Canvas(win, bg='lightgray', width=500, height=300)
c.place(x=0,y=100)

c.create_line(10,10,80,80,tag='hey bro')#(x1,y1,x2,y2)
c.create_oval(100,10,200,100)#円の描画
c.create_polygon(60,110,30,160,160,160)#三角形の描画
img=tk.PhotoImage(file='021_key.png')
c.create_image(230,130,image=img)
c.delete('hey bro')#'all'にすると全て消える。

def check():
    print('犬',v1.get(),'猫',v2.get())

v1=tk.BooleanVar()
v2=tk.BooleanVar()
v1.set(False)
v2.set(True)

cb1=tk.Checkbutton(win,text='犬',variable=v1)
cb2=tk.Checkbutton(win,text='猫',variable=v2)
cb1.place(x=20,y=0)
cb2.place(x=100,y=0)

btn2=tk.Button(win,text='Check',command=check)
btn2.place(x=100,y=100)

win.mainloop()#表示とイベント処理のループ

