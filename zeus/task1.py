import time
import socket
import numpy as np
#import data_test

## communication message
MSG_IMG_REQ = 1     # from robot to img (camera start request)
MSG_IMG_DATA = 2    # from img to robot (recognition data)
MSG_IMG_NO_DATA = 12    # from img to robot (no recognition)
MSG_GRAB_FAIL = 3   # from robot to img (grabbing failure)
MSG_FAIL_RES = 4    # from img to robot (response)
PROCESS_QUIT = 100

def call_task1(pipe, dot15Data):
    
    time.sleep(1)
    print('Robot task started..')
    m_out, m_in = pipe
    
    print('received Data(15 dots) = ',dot15Data)



    mode = 0 # 1:socket communication mode
    ##################### socket communication #####################
    if mode == 1:
        
        HOST = '192.168.0.23' #robot

        
        PORT = 9989

        
        # (address family) IPv4, TCP
        
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("OK")
        #client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        #client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)


        
        client_socket.connect((HOST, PORT))
        print('connected')
        print("Mad-Cow Activate")

        client_socket.sendall(dot15Data.encode())

    # wait for robot to move to 3d initial position
    time.sleep(5)

    #### 3d vision calibration ####
    # robot x
    X=np.mat([[0.09344
    ],
            [0.08786
    ],
            [0.30536
    ],
            [0.30146
    ],
            [0.19587]])

    # robot y
    Y=np.mat([[0.33936
    ],
            [0.64856
    ],
            [0.34328
    ],
            [0.65016
    ],
            [0.49506]])
    

    # image
    Prime=np.mat([[ 95, 94, 1],
                [504, 101, 1],
                [90, 376, 1],
                [499, 383, 1],
                [297, 238, 1]])

    dx_matrix = Prime.I*X
    dy_matrix = Prime.I*Y


    while True:
        

        m_in.send(MSG_IMG_REQ)
        print('Robot:image request has been sent..')

        data = m_in.recv()
        print('Robot has received data..: ', data)
        data_list = data.split()
        if len(data_list) >= 5 and float(data_list[8])<0.6 and float(data_list[8])>0.3: 
            print('center x - ', data_list[1],',center y - ',data_list[2],',theta - ',data_list[5],',depth - ',data_list[8])
                        
            image_x = float(data_list[1])
            image_y = float(data_list[2])
            
            # unit : mm
            dx = (dx_matrix[0, 0]*image_x + dx_matrix[1, 0]*image_y + dx_matrix[2, 0])*1000
            dy = (dy_matrix[0, 0]*image_x + dy_matrix[1, 0]*image_y + dy_matrix[2, 0])*1000
            # 441 = distance from camera to genga on plane, 103 = robot Z on plane
            dz = 441-float(data_list[8])*1000+103
            
            # with end tool tip
            # if dz<131.5:
            #     dz = 131.5

            
            
            # lebel
            if data_list[18] == 'jenga_up' and dz < 155:
                dz = 155
                
            if data_list[18] == 'jenga_up':
                data_list[17]=0

            
            if data_list[18] == 'jenga':
                # °ãÃÄÁ®ÀÖ´Â Á¨°¡ z value ÁöÁ¤ 
                if dz<119 and dz>109 and data_list[17]>10: #(0.425~0.435)
                    dz = 123
                elif dz<=112 and dz>87: #(0.435~0.456)
                    dz = 103

            # send (X, Y, Z, Rx, Ry, Rz, label) [mm, deg]
            ## dist = data_list[8]
            ## Rz = theta = data_list[5] 
            ## Ry = tilt = data_list[17] 
            result = str(dx)+ ' ' + str(dy) + ' ' + str(dz) + ' ' + data_list[5] +' '+ data_list[17]+' '+ str(180) +' '+ data_list[18]
            print('result : ', result)
            

            if mode == 1:
                client_socket.sendall(result.encode())
                
                data = client_socket.recv(1024)
                print('Received', repr(data.decode()))


        if(data == PROCESS_QUIT):
            print('process: call_task1 ends....')
            break

        

      