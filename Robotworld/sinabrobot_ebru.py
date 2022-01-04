from tkinter import *
import tkinter.font as tkFont
import socket
import time
from _thread import *
import threading



def send(socket):
    global go_send, signal
    while True:
        if go_send:
            socket.send(signal.encode())
            go_send = False
        else:
            if go_out:
                socket.close()
                exit()
            time.sleep(0.1)


def login():
    # 서버의 ip주소 및 포트
    ip = '192.168.137.102'
    port = 20002
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    client_socket.connect((ip, port))
    btn_pb.configure(image = img_pb2)
    btn_able_main()


    threading.Thread(target=send, args=(client_socket,)).start()
    threading.Thread(target=receive, args=(client_socket,)).start()
    exit()


def receive(socket):
    global size
    first = True
    while True:
        try:
            data = socket.recv(1024)
            if str(data.decode())=='Mission complete.':
                size = "None"
                if wind == "custom":
                    root2.destroy()
                elif wind == "pattern":
                    p_root2.destroy()
            chat_log['state'] = 'normal'
            if first:  # 이걸 처음 체크 이후 의미없이 매번 체크하므로 이렇게 하는 건 효율적이지 않음.
                chat_log.insert("end", str(data.decode()))
                first = False
            else:
                chat_log.insert("end", '\n' + str(data.decode()))
                chat_log.see('end')
            chat_log['state'] = 'disabled'
        except ConnectionAbortedError as e:
            chat_log['state'] = 'normal'
            chat_log.insert("end", '\n[System] 접속을 종료합니다.\n')
            chat_log['state'] = 'disabled'
            exit()

def try_login():
    global go_out
    start_new_thread(login,())
    login_button['state'] = 'disabled'
    logout_button['state'] = 'active'

    go_out = False

def try_logout():
    global go_out
    login_button['state'] = 'active'
    logout_button['state'] = 'disabled'

    btn_disable_main()

    btn_pb.configure(image = img_pb1)

    go_out = True


def btn_able_main():
    btn_gopattern['state'] = 'active';
    btn_custom['state'] = 'active';

def btn_disable_main():
    btn_gopattern['state'] = 'disabled';
    btn_custom['state'] = 'disabled';


def exit_ui():
    global signal, go_send
    signal = str(-1)
    go_send = True
    win.destroy()

def patternwindow():
    def patternA():
        global signal, go_send
        signal = str(3)
        go_send = True
        waitwindow_p()
    def patternB():
        global signal, go_send
        signal = str(4)
        go_send = True
        waitwindow_p()

    def flowerA():
        global signal, go_send
        signal = str(5)
        go_send = True
        waitwindow_p()

    def flowerB():
        global signal, go_send
        signal = str(6)
        go_send = True
        waitwindow_p()
    global wind
    wind = "pattern"
    global p_root
    p_root = Toplevel(win)
    # p_root.attributes('-fullscreen', True)
    p_root.geometry("1024x768")
    btn_exit1 = Button(p_root, image=img_exit1, command=p_root.destroy)
    btn_exit1.place(x=818, y=20)
    p_root.configure(bg='#282828')

    lab_logo = Label(p_root)
    # img=img.subsample(3)
    lab_logo.config(image=img_logo, bg='#282828')
    lab_logo.place(x=320, y=20)

    btn_flower1 = Button(p_root,command = flowerA, image=img_flower1,relief='flat')
    btn_flower1.place(x=170,y=180)
    btn_flower2 = Button(p_root, command = flowerB,image=img_flower2,relief='flat')
    btn_flower2.place(x=600, y=450)
    btn_patternA = Button(p_root, command=patternA,image=img_patternA,relief='flat')
    btn_patternA.place(x=600, y=180)
    btn_patternB = Button(p_root, command=patternB,image=img_patternB,relief='flat')
    btn_patternB.place(x=170, y=450)

    lab_cp = Label(p_root)
    lab_cp.config(text="Copyright © 2021 Sinabrobot. All Rights Reserved.", bg='#282828', fg='white',
                 font=tkFont.Font(family="Arial Black", size=10))
    lab_cp.place(x=340, y=740)


def customwindow():
    def btn_able():
        if size != "None":
            btn_red['state'] = 'active';
            btn_yel['state'] = 'active';
            btn_blue['state'] = 'active';
            btn_grn['state'] = 'active';
            btn_pp['state'] = 'active';
            btn_black['state'] = 'active';
            btn_white['state'] = 'active';
            #btn_skillA['state'] = 'active';
            #btn_skillB['state'] = 'active';
            #btn_skillC['state'] = 'active';
            #btn_skillD['state'] = 'active'

    def btn_disable():
        btn_red['state'] = 'disabled';
        btn_yel['state'] = 'disabled';
        btn_blue['state'] = 'disabled';
        btn_grn['state'] = 'disabled';
        btn_pp['state'] = 'disabled';
        btn_black['state'] = 'disabled';
        btn_white['state'] = 'disabled';
        #btn_skillA['state'] = 'disabled';
        #btn_skillB['state'] = 'disabled';
        #btn_skillC['state'] = 'disabled';
        #btn_skillD['state'] = 'disabled'

    def robot_home():
        global signal, go_send
        signal = str(1)

        go_send = True
        waitwindow_c()

    def robot_ready():
        global signal, go_send
        signal = str(2)

        go_send = True
        waitwindow_c()

    def pick_big():
        global size
        size = "Big"
        size_select()

    def pick_small():
        global size
        size = "Small"
        size_select()
    def robot_skillA():
        global signal, go_send
        signal = str(7)

        go_send = True
        waitwindow_c()
    def robot_skillB():
        global signal, go_send
        signal = str(8)

        go_send = True
        waitwindow_c()
    def robot_skillC():
        global signal, go_send
        signal = str(9)

        go_send = True
        waitwindow_c()
    def robot_skillD():
        global signal, go_send
        signal = str(10)

        go_send = True
        waitwindow_c()
    def robot_red():
        global signal, go_send
        if size == "Big":
            signal = str(11)
        elif size == "Small":
            signal = str(21)
        else:
            Big_button['state'] = 'active'
            Small_button['state'] = 'active'
        go_send = True
        waitwindow_c()

    def robot_yel():
        global signal, go_send
        if size == "Big":
            signal = str(12)
        elif size == "Small":
            signal = str(22)
        else:
            Big_button['state'] = 'active'
            Small_button['state'] = 'active'
        go_send = True
        waitwindow_c()

    def robot_blue():
        global signal, go_send
        if size == "Big":
            signal = str(13)
        elif size == "Small":
            signal = str(23)
        else:
            Big_button['state'] = 'active'
            Small_button['state'] = 'active'
        go_send = True
        waitwindow_c()
    def robot_grn():
        global signal, go_send
        if size == "Big":
            signal = str(14)
        elif size == "Small":
            signal = str(24)
        else:
            Big_button['state'] = 'active'
            Small_button['state'] = 'active'
        go_send = True
        waitwindow_c()
    def robot_pp():
        global signal, go_send
        if size == "Big":
            signal = str(15)
        elif size == "Small":
            signal = str(25)
        else:
            Big_button['state'] = 'active'
            Small_button['state'] = 'active'
        go_send = True
        waitwindow_c()

    def robot_black():
        global signal, go_send
        if size == "Big":
            signal = str(16)
        elif size == "Small":
            signal = str(26)
        else:
            Big_button['state'] = 'active'
            Small_button['state'] = 'active'
        go_send = True
        waitwindow_c()

    def robot_white():
        global signal, go_send
        if size == "Big":
            signal = str(17)
        elif size == "Small":
            signal = str(27)
        else:
            Big_button['state'] = 'active'
            Small_button['state'] = 'active'
        go_send = True
        waitwindow_c()

    def size_select():
        if size == "Big":
            Big_button['state'] = 'disabled'
            Small_button['state'] = 'active'
            btn_able()
        elif size == "Small":
            Big_button['state'] = 'active'
            Small_button['state'] = 'disabled'
            btn_able()
        else:
            Big_button['state'] = 'active'
            Small_button['state'] = 'active'
            btn_disable()


    global wind
    wind = "custom"
    global root
    root = Toplevel(win)
    # root.attributes('-fullscreen', True)
    root.geometry("1024x768")
    btn_exit = Button(root,image=img_exit2,command=root.destroy)
    btn_exit.place(x=818, y=534)
    root.configure(bg='#282828')




    lab_d = Label(root)
    # img=img.subsample(3)
    lab_d.config(image=img_logo, bg='#282828')
    lab_d.place(x=320, y=20)

    lab_c = Label(root)
    lab_c.config(text="Copyright © 2021 Sinabrobot. All Rights Reserved.", bg='#282828', fg='white',
                 font=tkFont.Font(family="Arial Black", size=10))
    lab_c.place(x=340, y=740)

    Big_button = Button(root, image = img_bigbtn, command=pick_big, relief='flat');
    Big_button.place(x=30, y=140)
    Small_button = Button(root, image = img_smallbtn, command=pick_small,relief='flat')
    Small_button.place(x=227, y=140)

    lab_s = Label(root)
    lab_s.config(image=img_s,bg='#282828')
    lab_s.place(x=630, y=540)

    btn_red = Button(root, image=img_redbtn,command = robot_red, bg='white', relief='flat')
    btn_red.place(x=30, y=337)


    btn_yel = Button(root, image=img_yelbtn, command = robot_yel,bg='white', relief='flat')
    btn_yel.place(x=227, y=337)

    btn_blue = Button(root, image=img_bluebtn,command = robot_blue, bg='white', relief='flat')
    btn_blue.place(x=424, y=337)

    btn_grn = Button(root, image=img_grnbtn,command = robot_grn, bg='white', relief='flat')
    btn_grn.place(x=30, y=534)

    btn_pp = Button(root, image=img_ppbtn,command = robot_pp, bg='white', relief='flat')
    btn_pp.place(x=227, y=534)

    btn_black = Button(root, image=img_blackbtn,command = robot_black, bg='white', relief='flat')
    btn_black.place(x=424, y=534)

    btn_white = Button(root, image=img_whitebtn,command = robot_white, bg='white', relief='flat')
    btn_white.place(x=424, y=140)
    # btn_white.place(x=30, y=564)

    btn_skillA = Button(root, image=img_skillA, command=robot_skillA, bg='white', relief='flat')
    btn_skillA.place(x=621, y=140)

    btn_skillB = Button(root, image=img_skillB, command=robot_skillB, bg='white', relief='flat')
    btn_skillB.place(x=818, y=140)

    btn_skillC = Button(root, image=img_skillC, command=robot_skillC,bg='white', relief='flat')
    btn_skillC.place(x=621, y=337)

    btn_skillD = Button(root, image=img_skillD, command=robot_skillD, bg='white', relief='flat')
    btn_skillD.place(x=818, y=337)

    btn_disable()

def waitwindow_c():

    def btn_disable():
        btn_red['state'] = 'disabled';
        btn_yel['state'] = 'disabled';
        btn_blue['state'] = 'disabled';
        btn_grn['state'] = 'disabled';
        btn_pp['state'] = 'disabled';
        btn_black['state'] = 'disabled';
        btn_white['state'] = 'disabled';
        btn_skillA['state'] = 'disabled';
        btn_skillB['state'] = 'disabled';
        btn_skillC['state'] = 'disabled';
        btn_skillD['state'] = 'disabled'


    def destroy_rt2():
        global root, root2
        root.destroy()
        root2.destroy()

    global root2
    root2 = Toplevel(win)
    # root2.attributes('-fullscreen', True)
    root2.geometry("1024x768")
    btn_exit2 = Button(root2,image=img_exit2,command=destroy_rt2)
    btn_exit2.place(x=818, y=534)
    root2.configure(bg='#282828')



    lab_d = Label(root2)
    # img=img.subsample(3)
    lab_d.config(image=img_logo, bg='#282828')
    lab_d.place(x=320, y=20)

    lab_c = Label(root2)
    lab_c.config(text="Copyright © 2021 Sinabrobot. All Rights Reserved.", bg='#282828', fg='white',
                 font=tkFont.Font(family="Arial Black", size=10))
    lab_c.place(x=340, y=740)

    Big_button = Button(root2, image = img_bigbtn,state ='disabled',  relief='flat')
    Big_button.place(x=30, y=140)
    Small_button = Button(root2, image = img_smallbtn,state='disabled', relief='flat')
    Small_button.place(x=227, y=140)

    lab_s = Label(root2)
    lab_s.config(image=img_s,bg='#282828')
    lab_s.place(x=630, y=540)

    btn_red = Button(root2, image=img_redbtn, bg='white', relief='flat')
    btn_red.place(x=30, y=337)


    btn_yel = Button(root2, image=img_yelbtn, bg='white', relief='flat')
    btn_yel.place(x=227, y=337)

    btn_blue = Button(root2, image=img_bluebtn, bg='white', relief='flat')
    btn_blue.place(x=424, y=337)

    btn_grn = Button(root2, image=img_grnbtn, bg='white', relief='flat')
    btn_grn.place(x=30, y=534)

    btn_pp = Button(root2, image=img_ppbtn, bg='white', relief='flat')
    btn_pp.place(x=227, y=534)

    btn_black = Button(root2, image=img_blackbtn, bg='white', relief='flat')
    btn_black.place(x=424, y=534)

    btn_white = Button(root2, image=img_whitebtn, bg='white', relief='flat')
    btn_white.place(x=424, y=140)
    # btn_white.place(x=30, y=564)

    btn_skillA = Button(root2, image=img_skillA, bg='white', relief='flat')
    btn_skillA.place(x=621, y=140)

    btn_skillB = Button(root2, image=img_skillB, bg='white', relief='flat')
    btn_skillB.place(x=818, y=140)

    btn_skillC = Button(root2, image=img_skillC, bg='white', relief='flat')
    btn_skillC.place(x=621, y=337)

    btn_skillD = Button(root2, image=img_skillD, bg='white', relief='flat')
    btn_skillD.place(x=818, y=337)

    btn_disable()

def waitwindow_p():
    def btn_disable_p():
        btn_flower1['state'] = 'disabled'; btn_flower2['state'] = 'disabled';btn_patternA['state'] = 'disabled';btn_patternB['state'] = 'disabled'
    def destroy_prt2():
        global p_root, p_root2
        p_root.destroy()
        p_root2.destroy()

    global p_root2
    p_root2 = Toplevel(win)
    p_root2.attributes('-fullscreen', True)
    # p_root.geometry("1024x768")
    btn_exit1 = Button(p_root2, image=img_exit1, command=destroy_prt2)
    btn_exit1.place(x=818, y=20)
    p_root2.configure(bg='#282828')

    lab_logo = Label(p_root2)
    # img=img.subsample(3)
    lab_logo.config(image=img_logo, bg='#282828')
    lab_logo.place(x=320, y=20)

    btn_flower1 = Button(p_root2,  image=img_flower1, relief='flat')
    btn_flower1.place(x=170, y=180)
    btn_flower2 = Button(p_root2,  image=img_flower2, relief='flat')
    btn_flower2.place(x=600, y=450)
    btn_patternA = Button(p_root2,  image=img_patternA, relief='flat')
    btn_patternA.place(x=600, y=180)
    btn_patternB = Button(p_root2,  image=img_patternB, relief='flat')
    btn_patternB.place(x=170, y=450)

    lab_cp = Label(p_root2)
    lab_cp.config(text="Copyright © 2021 Sinabrobot. All Rights Reserved.", bg='#282828', fg='white',
                  font=tkFont.Font(family="Arial Black", size=10))
    lab_cp.place(x=340, y=740)

    btn_disable_p()


go_out, go_send = False, False
size = "None"
wind = "None"
signal = str(0)
win = Tk()
win.title("Sinabro_Ebru")

# win.attributes('-fullscreen', True) # 전체화면
win.geometry("1024x768")
win.configure(bg='#282828')
font = tkFont.Font(family="Arial Black", size=16, weight="bold")
login_button = Button(win,text="Log in", width=8 , command=try_login, bg ='white',relief = 'flat', font=font); login_button.place(x=840, y=20)
logout_button = Button(win,text = "Log out", width=8 ,state = 'disabled', command = try_logout, font=font,bg='white',relief = 'flat'); logout_button.place(x=840, y=80)


img_logo = PhotoImage(file="Sinabrobot_Logo3.png", master=win)
img_logo = img_logo.subsample(2)
img_bigbtn = PhotoImage(file="big.png",master=win)
img_smallbtn = PhotoImage(file="small.png",master=win)
img_redbtn = PhotoImage(file="빨간버튼.png",master=win)
img_yelbtn = PhotoImage(file="노란버튼.png",master=win)
img_bluebtn = PhotoImage(file="파란버튼.png",master=win)
img_grnbtn = PhotoImage(file="초록버튼.png",master=win)
img_ppbtn = PhotoImage(file="보라버튼.png",master=win)
img_blackbtn = PhotoImage(file="검정버튼.png",master=win)
img_whitebtn = PhotoImage(file="하얀버튼.png",master=win)
img_skillA = PhotoImage(file="스킬1.png",master=win)
img_skillB = PhotoImage(file="스킬2.png",master=win)
img_skillC = PhotoImage(file="스킬3.png",master=win)
img_skillD = PhotoImage(file="스킬4.png",master=win)
img_patternA = PhotoImage(file="pattern1.png",master=win)
#img_patternA = img_patternA.subsample(2)
img_patternB = PhotoImage(file="pattern2.png",master=win)
#img_patternB = img_patternB.subsample(2)
img_gopattern = PhotoImage(file="go_pattern.png",master=win)
img_custom = PhotoImage(file="custom.png",master=win)
img_exit1 = PhotoImage(file="exit1.png",master = win)
img_exit2 = PhotoImage(file="exit2.png",master = win)
img_s = PhotoImage(file="s.png", master = win)
img_flower1 = PhotoImage(file="flower1.png", master = win)
#img_flower1 = img_flower1.subsample(2)
img_flower2 = PhotoImage(file="flower2.png", master = win)
#img_flower2 = img_flower2.subsample(2)
img_pb1 = PhotoImage(file="pb1.png", master = win)
img_pb2 = PhotoImage(file="pb2.png", master = win)
#lab_a = Label(win)
#lab_a.config(text='connect with robot not yet')
#lab_a.place(x=0, y=0)
"""채팅 로그"""
chat_frame = Frame(win)
scrollbar = Scrollbar(chat_frame);
scrollbar.pack(side='right', fill='y')

chat_log = Text(chat_frame, width=52, height=8, state='disabled', yscrollcommand=scrollbar.set,font=tkFont.Font(family="Arial", size=11))
chat_log.pack(side='left')  # place(x=20, y=60)
scrollbar['command'] = chat_log.yview
# chat_frame.place(x=30, y=20)
"""로고 삽입"""
lab_d = Label(win)

lab_d.config(image=img_logo, bg='#282828')
lab_d.place(x=320, y=20)

lab_c = Label(win)
lab_c.config(text="Copyright © 2021 Sinabrobot. All Rights Reserved.",bg='#282828', fg='white', font=tkFont.Font(family="Arial Black", size=10))
lab_c.place(x=340, y=740)

# img_home = PhotoImage(file="robot_home.png",master=win)
# img_home=img_home.subsample(4)




btn_gopattern = Button(win, image = img_gopattern, command=patternwindow,relief = 'flat')
btn_gopattern.place(x=20,y=140)

btn_custom = Button(win, image = img_custom,relief = 'flat',command=customwindow)
btn_custom.place(x= 530,y=140)

btn_pb = Button(win, image=img_pb1, command=exit_ui, bg='#282828',relief = 'flat')
btn_pb.place(x=460, y=630)

# btn_disable_main()

win.mainloop()