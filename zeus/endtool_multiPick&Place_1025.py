#!/usr/bin/python
# -*- coding: utf-8 -*-
## 1. 초기 설정① #######################################
# 라이브러리 가져오기
## 1．초기 설정 ①　모듈 가져오기 ######################
import socket

from i611_MCS import * # 로봇제어 기본기능
from teachdata import * # 교시데이터 사용
from i611_extend import * # 확장기능 사용 (pallet)
from rbsys import * # 관리 프로그램 사용
from i611_common import * # i611Robot 클래스 메소드의 예외처리
from i611_io import * # I/O 신호를 제어
from i611shm import *  # 공유메모리에 액세스
import time # 시간관련 모듈
import pdb # 1줄씩 확인 모듈

from socket import *
from thread import *

import math


## 건드려도 됨
def jengaPick(rb, pick_x, pick_y, pick_z, pick_rz, pick_ry, pick_rx,label):

    print('pick_x, pick_y, pick_z, pick_rz, pick_ry, pick_rx :',pick_x, pick_y, pick_z, pick_rz, pick_ry, pick_rx)

    if pick_ry>20:
        pick_ry = 20
    elif pick_ry>0 and pick_ry<10:
        pick_ry = 0

    # set minimum z
    if label == 'jenga':
        pick_z = 94

    elif label == 'jenga_side':
        pick_z = 98
    elif label == 'jenga_up' and pick_z < 155:
        pick_z = 155


    pick_up_z = 200

    # example
    #test3 = Position(-86+104*math.cos(math.pi * 57 / 180), 534+104*math.sin(math.pi * 57 / 180), 250, 57, 0, -180)

    ######### 젠가 집을 때 - 숫자가 크면 더 내려감, 젠가 사이드 안맞을 수 있음
    # if label == 'jenga':
    #     pick_z -= 18#18
    # elif label == 'jenga_side':
    #     pick_z -= 12


    print('new_pick_z : ', pick_z)
    pick_up = Position(pick_x-104*math.cos(math.pi * pick_rz / 180), pick_y-104*math.sin(math.pi * pick_rz / 180), pick_up_z, pick_rz, pick_ry-45, pick_rx)
    pick_down = Position(pick_x-104*math.cos(math.pi * pick_rz / 180), pick_y-104*math.sin(math.pi * pick_rz / 180), pick_z, pick_rz, pick_ry-45, pick_rx)

    if label == 'jenga' and pick_z == 123:
        pick_z = 123
        pick_up = Position(pick_x - 147*math.sin(45-pick_ry) * math.cos(math.pi * pick_rz / 180),
                           pick_y - 147*math.sin(45-pick_ry) * math.sin(math.pi * pick_rz / 180), pick_up_z, pick_rz, pick_ry - 45, pick_rx)
        pick_down = Position(pick_x - 147*math.sin(45-pick_ry) * math.cos(math.pi * pick_rz / 180),
                             pick_y - 147*math.sin(45-pick_ry) * math.sin(math.pi * pick_rz / 180), pick_z, pick_rz, pick_ry - 45, pick_rx)


    dout(48, '100') # 그리퍼 open
    time.sleep(0.2)
    # 로봇관절 풀어주기

    initialPos = Joint(0, 0, -90, 0, -90, 0)
    rb.move(initialPos)

    rb.move(pick_up)  # 픽업 오프셋 이동
    print("1번 위치 이동")
    time.sleep(0.2)  # 도달 후 잠시 대기

    speed = 10
    motion = MotionParam(jnt_speed=speed, lin_speed=speed, pose_speed=speed, acctime=0.4, dacctime=0.4, overlap=20)
    rb.motionparam(motion)
    #pos01 = rb.getjnt()

    # rb.changetool(1)
    # rb.toolmove(dry=45)
    # #rb.toolmove(dz=-64)
    #
    # rb.changetool(0)
    #current = rb.getpos()
    #pick_down = current.shift(dz=-64)
    rb.move(pick_down)  # 픽업위치 이동
    print("2번 위치 이동")
    dout(48, '000')  # Arm I/O 초기화
    time.sleep(0.8)
    dout(48, '001')  # 그리퍼 close
    print("오브젝트 픽업완료")
    time.sleep(0.2)
    speed = 40
    motion = MotionParam(jnt_speed=speed, lin_speed=speed, pose_speed=speed, acctime=0.4, dacctime=0.4, overlap=20)
    rb.motionparam(motion)
    rb.line(pick_up)  # 픽업 오프셋 이동
    print("jengaPick end..")

def jengaPlace(rb, place_x, place_y, recievedTheta):

    time.sleep(0.2)
    
    # 로봇의 position 계산
    # dist_robot_to_endtip = 114 #[mm] 110
    # theta = 90 - recievedTheta # theta : 이미지를 똑바로 놨을때 접선의 기울기(0도~180도) ,receivedData:-90~90 => 보정해줌
    # newX = place_x-dist_robot_to_endtip*math.cos(math.pi*theta/180)
    # newY = place_y-dist_robot_to_endtip*math.sin(math.pi*theta/180)
    #
    # if theta>90 and theta<180:
    #     theta = theta-180
    #     newX = place_x-dist_robot_to_endtip*math.cos(math.pi*abs(theta)/180)
    #     newY = place_y+dist_robot_to_endtip*math.sin(math.pi*abs(theta)/180)
    test3 = Position(-86+104*math.cos(math.pi * 57 / 180), 534+104*math.sin(math.pi * 57 / 180), 250, 57, 0, -180)

    # print('newX, newY, rz = ', newX, newY, theta)

    # 로봇관절 풀어주기

    initialPos = Joint(0, 0, -90, 0, -90, 0)
    #p = rb.Joint2Position(initialPos)

    #rb.move(initialPos)
    rb.home()

    #0~90
    # if recievedTheta>0 and recievedTheta<=90:
    #     newTheta = -recievedTheta+90
    # else:
    #     newTheta = -recievedTheta+90

    newTheta = -recievedTheta + 90 # recievedTheta, recievedTheta+90, -recievedTheta, recievedTheta-90
    test_place_up = Position(place_x + 104 * math.cos(math.pi * (newTheta) / 180),
                             place_y + 104 * math.sin(math.pi * (newTheta) / 180), 220,
                             int(newTheta), 45, -180)

    test_place_down = Position(place_x + 104 * math.cos(math.pi * (newTheta) / 180),
                               place_y + 104 * math.sin(math.pi * (newTheta) / 180), 132,
                               int(newTheta), 45, -180)

    time.sleep(0.2)

    rb.move(test_place_up) # 플레이스 위치 상공으로 이동
    time.sleep(0.2)
    speed = 20
    motion = MotionParam(jnt_speed=speed, lin_speed=speed, pose_speed=speed, acctime=0.4, dacctime=0.4, overlap=20)
    rb.motionparam(motion)

    rb.line(test_place_down) # 플레이스 위치로 이동
    dout(48, '000')  # Arm I/O 초기화
    time.sleep(0.2)
    dout(48,'100') # 그리퍼 open
    time.sleep(0.2)
    speed = 40
    motion = MotionParam(jnt_speed=speed, lin_speed=speed, pose_speed=speed, acctime=0.4, dacctime=0.4, overlap=20)
    rb.motionparam(motion)
    rb.line(test_place_up) # 플레이스 위치 상공으로 이동


def jenga_sidePlace(rb, place_x, place_y, recievedTheta):
    time.sleep(0.2)

    # 로봇관절 풀어주기
    initialPos = Joint(0, 0, -90, 0, -90, 0)
    #rb.move(initialPos)
    rb.home()

    newTheta = -recievedTheta
    # if recievedTheta > 0 and recievedTheta < 90:
    #     newTheta = -recievedTheta #-recievedTheta+90
    #
    # else:
    #     newTheta = -recievedTheta

    print('recievedTheta:',recievedTheta)

    test_place_up = Position(place_x + 104 * math.cos(math.pi * (newTheta) / 180), place_y + 104 * math.sin(math.pi * (newTheta) / 180), 220, int(newTheta), 45, -180)
    test_place_down = Position(place_x + 104 * math.cos(math.pi * (newTheta) / 180), place_y + 104 * math.sin(math.pi * (newTheta) / 180), 132, int(newTheta), 45, -180)



    try:
        time.sleep(0.2)
        rb.move(test_place_up)  # 플레이스 위치 상공으로 이동
        print('test_place_up finished..')
        time.sleep(0.2)
        speed = 10
        motion = MotionParam(jnt_speed=speed, lin_speed=speed, pose_speed=speed, acctime=0.4, dacctime=0.4, overlap=20)
        rb.motionparam(motion)
        rb.line(test_place_down)  # 플레이스 위
        # 치로 이동
        print('test_place_down finished..')
        dout(48, '000')  # Arm I/O 초기화
        time.sleep(0.2)
        dout(48, '100')  # 그리퍼 open
        time.sleep(0.2)
        speed = 40
        motion = MotionParam(jnt_speed=speed, lin_speed=speed, pose_speed=speed, acctime=0.4, dacctime=0.4, overlap=20)
        rb.motionparam(motion)
        rb.line(test_place_up)  # 플레이스 위치 상공으로 이동

    except Robot_error:
        print
        "Robot_error"
        #dout(48, '000')
        if recievedTheta > 0 and recievedTheta < 90:
            test_place_up = Position(place_x - 104 * math.cos(math.pi * (-recievedTheta+90) / 180),
                                     place_y - 104 * math.sin(math.pi * (-recievedTheta+90) / 180), 220,
                                     int(-recievedTheta+90), 45+90,0)
            test_place_down = Position(place_x - 104 * math.cos(math.pi * (-recievedTheta+90) / 180),
                                       place_y - 104 * math.sin(math.pi * (-recievedTheta+90) / 180), 130,
                                       int(-recievedTheta+90), 45+90, 0)

        time.sleep(0.2)
        rb.move(test_place_up)  # 플레이스 위치 상공으로 이동
        print('test_place_up finished..')
        time.sleep(0.2)
        rb.line(test_place_down)  # 플레이스 위
        # 치로 이동
        print('test_place_down finished..')
        dout(48, '000')  # Arm I/O 초기화
        time.sleep(0.2)
        dout(48, '100')  # 그리퍼 open
        time.sleep(0.2)
        rb.line(test_place_up)  # 플레이스 위치 상공으로 이동


# unreachable 유발자임/
def jenga_upPlace(rb, place_x, place_y, recievedTheta):
    time.sleep(0.2)

    # 로봇관절 풀어주기
    initialPos = Joint(0, 0, -90, 0, -90, 0)
    rb.move(initialPos)


    if recievedTheta>0 and recievedTheta<=90:
        newTheta = 90-recievedTheta

    else:
        newTheta = recievedTheta

    test_place_up = Position(place_x - 104 * math.cos(math.pi * newTheta / 180), place_y - 104 * math.sin(math.pi * newTheta / 180), 200, int(newTheta), -45, -180)
    test_place_down = Position(place_x - 104 * math.cos(math.pi * newTheta / 180), place_y - 104 * math.sin(math.pi * newTheta / 180), 150, int(newTheta), -45, -180)

    rb.move(test_place_up)  # 플레이스 위치 상공으로 이동
    rb.line(test_place_down)  # 플레이스 위치로 이동
    dout(48, '000')  # Arm I/O 초기화
    time.sleep(0.2)
    dout(48, '100')  # 그리퍼 open
    time.sleep(0.2)
    rb.line(test_place_up)  # 플레이스 위치 상공으로 이동


def jengaKnock(rb, place_x, place_y, recievedTheta):

    dout(48, '000')  # Arm I/O 초기화
    time.sleep(0.2)
    dout(48, '001')  # 그리퍼 close

    dist = 150
    if recievedTheta>0 and recievedTheta<90:
        ready_up = Position(place_x-150*math.cos(math.pi *recievedTheta/180), place_y-150*math.sin(math.pi *recievedTheta/180), 200, int(recievedTheta), 0, -180)
        ready_down = Position(place_x-150*math.cos(math.pi *recievedTheta/180), place_y-150*math.sin(math.pi *recievedTheta/180), 160, int(recievedTheta), 0, -180)
    else:
        # need to be revised
        ready_up = Position(place_x + 100 * math.cos(math.pi *recievedTheta/180), place_y + 100 * math.sin(math.pi *recievedTheta/180),200,
                            int(recievedTheta), 0, -180)
        ready_down = Position(place_x + 100 * math.cos(math.pi *recievedTheta/180), place_y + 100 * math.sin(math.pi *recievedTheta/180),160,
                              int(recievedTheta), 0, -180)

    knock = Position(place_x,place_y,160,int(recievedTheta), 0, -180)
    rb.move(ready_up)  # 플레이스 위치 상공으로 이동
    time.sleep(0.2)
    rb.move(ready_down)
    time.sleep(0.2)
    rb.line(knock)

    print('program finished !! :) bye bye ~ ')


def getPositionData(data):
    # data from 3d vision(X,Y,Z,rz,ry,rx)
    positionData = []
    print('data===',data)
    # save received data to positionData list
    for i in range(6):
        print(data.split()[i].decode())
        positionData.append(int(float(data.split()[i].decode())))

    return positionData

def getLabel(data):
    # label
    print(data.split()[6].decode())
    label = data.split()[6].decode()

    return label

# 이 함수 호출하면 s자곡선 테스트할 수 있음
def testSline(rb):
    #### s자 곡선 테스트 ####
    # 얘네 이상함 90바꿔줘야함
    # jengaPlace(rb, 41, 546, 6)
    jenga_sidePlace(rb, 41, 546, 6)#=>unreachable
    # jenga_upPlace(rb, 41, 546, 6)
    # print('1번 완료..')

    # jengaPlace(rb, 38, 515, 28)
    jenga_sidePlace(rb, 38, 515, 28)
    # jenga_upPlace(rb, 38, 515, 28)
    # print('2번 완료..')

    # jengaPlace(rb, 15,481,49)
    jenga_sidePlace(rb, 15,481,49)
    # jenga_upPlace(rb, 15, 481, 49)
    # print('3번 완료..')

    # jengaPlace(rb,-19,461,74)
    jenga_sidePlace(rb, -19,461,74)
    # jenga_upPlace(rb, -19, 461, 74)
    # print('4번 완료..')

    # jengaPlace(rb,-59,457,-79)
    jenga_sidePlace(rb, -59,457,-79)
    # jenga_upPlace(rb, -59,457,-79)
    # print('5번 완료..')

    # jengaPlace(rb,-95,473,-51) # 얘 이상함
    jenga_sidePlace(rb, -95,473,-51)
    # jenga_upPlace(rb, -95,473,-51)
    # print('6번 완료..')

    # jengaPlace(rb,-122,506,-25)
    jenga_sidePlace(rb, -122,506,-25)
    # jenga_upPlace(rb, -122,506,-25)
    # print('7번 완료..')
    #
    #
    # jengaPlace(rb,-131,541,-12)
    jenga_sidePlace(rb, -131,541,-12)
    # jenga_upPlace(rb, -131,541,-12)
    # print('8번 완료..')

    # jengaPlace(rb,-139,576,-21)
    jenga_sidePlace(rb, -139,576,-21)
    # jenga_upPlace(rb, -139,576,-21)
    # print('9번 완료..')

    # jengaPlace(rb,-162,610,-44)
    jenga_sidePlace(rb, -162,610,-44)
    # jenga_upPlace(rb, -162,610,-44)
    # print('10번 완료..')
    #
    #
    # jengaPlace(rb,-197,630,-72)
    jenga_sidePlace(rb, -197,630,-72)
    # jenga_upPlace(rb, -197,630,-72)
    # print('11번 완료..')
    #
    #
    # jengaPlace(rb,-232,631,-97)
    jenga_sidePlace(rb, -232,631,-97)
    # jenga_upPlace(rb, -232,631,-97)
    # print('12번 완료..')

    # jengaPlace(rb,-267,618,59)
    jenga_sidePlace(rb, -267,618,59)#왜 얘만 이상함?????????
    # jenga_upPlace(rb, -267,618,59)
    # print('13번 완료..')

    # jengaPlace(rb,-296,590,40)
    jenga_sidePlace(rb, -296,590,40)
    # jenga_upPlace(rb, -296,590,40)
    # print('14번 완료..')

    # jengaPlace(rb,-308,555,20)
    jenga_sidePlace(rb, -308, 555, 20)
    # jenga_upPlace(rb, -308,555,20)
    # print('15번 완료..')

    jengaKnock(rb, -308, 555, 20)

# 이 함수 호출하면 s자곡선 테스트할 수 있음
def testWline(rb):
    #### s자 곡선 테스트 ####
    # 얘네 이상함 90바꿔줘야함
    # jengaPlace(rb, 41, 546, 6)
    # jenga_sidePlace(rb,-24,627,-32)#=>unreachable
    # # jenga_upPlace(rb, 41, 546, 6)
    # # print('1번 완료..')
    #
    # # jengaPlace(rb, 38, 515, 28)
    # jenga_sidePlace(rb, -1, 593, -24)
    # # jenga_upPlace(rb, 38, 515, 28)
    # # print('2번 완료..')
    #
    # # jengaPlace(rb, 15,481,49)
    # jenga_sidePlace(rb, 25,551,-15)
    # # jenga_upPlace(rb, 15, 481, 49)
    # # print('3번 완료..')
    #
    # # jengaPlace(rb,-19,461,74)
    # jenga_sidePlace(rb, 28,509,17)
    # # jenga_upPlace(rb, -19, 461, 74)
    # # print('4번 완료..')
    #
    # # jengaPlace(rb,-59,457,-79)
    # jenga_sidePlace(rb, 3,466,47)
    # # jenga_upPlace(rb, -59,457,-79)
    # # print('5번 완료..')
    #
    # # jengaPlace(rb,-95,473,-51) # 얘 이상함
    # jenga_sidePlace(rb, -34,443,-89)
    # # jenga_upPlace(rb, -95,473,-51)
    # # print('6번 완료..')
    #
    # # jengaPlace(rb, -76,467,-64)
    # jenga_sidePlace(rb, -76,467,-64)
    # # jenga_upPlace(rb, -122,506,-25)
    # # print('7번 완료..')
    # #
    # #
    # # jengaPlace(rb,-131,541,-12)
    # jenga_sidePlace(rb, -117,482,97)
    # # jenga_upPlace(rb, -131,541,-12)
    # # print('8번 완료..')
    #
    # # jengaPlace(rb,-139,576,-21)
    # jenga_sidePlace(rb, -155,452,57)
    # # jenga_upPlace(rb, -139,576,-21)
    # # print('9번 완료..')
    #
    # # jengaPlace(rb,-162,610,-44)
    # jenga_sidePlace(rb, -197,425,-96)
    # # jenga_upPlace(rb, -162,610,-44)
    # # print('10번 완료..')
    # #
    # #
    # # jengaPlace(rb,-197,630,-72)
    # jenga_sidePlace(rb, -238,438,-63)
    # # jenga_upPlace(rb, -197,630,-72)
    # # print('11번 완료..')
    # #
    # #
    # # jengaPlace(rb,-232,631,-97)
    # jenga_sidePlace(rb, -279,468,-30)
    # # jenga_upPlace(rb, -232,631,-97)
    # # print('12번 완료..')

    # jengaPlace(rb,-267,618,59)
    jenga_sidePlace(rb, -287,510,83)#왜 얘만 이상함?????????
    # jenga_upPlace(rb, -267,618,59)
    # print('13번 완료..')

    # # jengaPlace(rb,-296,590,40)
    # jenga_sidePlace(rb, -274,552,26)
    # # jenga_upPlace(rb, -296,590,40)
    # # print('14번 완료..')
    #
    # # jengaPlace(rb,-308,555,20)
    # jenga_sidePlace(rb, -253, 595, 30)
    # # jenga_upPlace(rb, -308,555,20)
    # # print('15번 완료..')

    jengaKnock(rb,-253, 595, 30)



def testUline(rb):
    #### s자 곡선 테스트 ####
    # 얘네 이상함 90바꿔줘야함
    # jengaPlace(rb, 41, 546, 6)
    jenga_sidePlace(rb,-44,488,21)#=>unreachable
    # jenga_upPlace(rb, 41, 546, 6)
    # print('1번 완료..')

    # jengaPlace(rb, 38, 515, 28)
    jenga_sidePlace(rb, -37, 508, 13)
    # jenga_upPlace(rb, 38, 515, 28)
    # print('2번 완료..')

    # jengaPlace(rb, 15,481,49)
    jenga_sidePlace(rb, -34,536,6)
    # jenga_upPlace(rb, 15, 481, 49)
    # print('3번 완료..')

    # jengaPlace(rb,-19,461,74)
    jenga_sidePlace(rb, -33,563,-90)
    # jenga_upPlace(rb, -19, 461, 74)
    # print('4번 완료..')

    # jengaPlace(rb,-59,457,-79)
    jenga_sidePlace(rb, -36,590,-18)
    # jenga_upPlace(rb, -59,457,-79)
    # print('5번 완료..')

    # jengaPlace(rb,-95,473,-51) # 얘 이상함
    jenga_sidePlace(rb, -52,613,-50)
    # jenga_upPlace(rb, -95,473,-51)
    # print('6번 완료..')

    # jengaPlace(rb,-122,506,-25)
    jenga_sidePlace(rb, -79,624,-78)
    # jenga_upPlace(rb, -122,506,-25)
    # print('7번 완료..')
    #
    #
    # jengaPlace(rb,-131,541,-12)
    jenga_sidePlace(rb, -106,623,-96)
    # jenga_upPlace(rb, -131,541,-12)
    # print('8번 완료..')

    # jengaPlace(rb,-139,576,-21)
    jenga_sidePlace(rb, -133,615,67)
    # jenga_upPlace(rb, -139,576,-21)
    # print('9번 완료..')

    # jengaPlace(rb,-162,610,-44)
    jenga_sidePlace(rb, -160,598,52)
    # jenga_upPlace(rb, -162,610,-44)
    # print('10번 완료..')
    #
    #
    # jengaPlace(rb,-197,630,-72)
    jenga_sidePlace(rb, -183,574,-41)
    # jenga_upPlace(rb, -197,630,-72)
    # print('11번 완료..')
    #
    #
    # jengaPlace(rb,-232,631,-97)
    jenga_sidePlace(rb, -202,546,34)
    # jenga_upPlace(rb, -232,631,-97)
    # print('12번 완료..')

    # jengaPlace(rb,-267,618,59)
    jenga_sidePlace(rb, -218,518,30)#왜 얘만 이상함?????????
    # jenga_upPlace(rb, -267,618,59)
    # print('13번 완료..')

    # jengaPlace(rb,-296,590,40)
    jenga_sidePlace(rb, -233,490,34)
    # jenga_upPlace(rb, -296,590,40)
    # print('14번 완료..')

    # jengaPlace(rb,-308,555,20)
    jenga_sidePlace(rb, -249, 470, 39)
    # jenga_upPlace(rb, -308,555,20)
    # print('15번 완료..')

    jengaKnock(rb,  -249, 470, 39)


def main():
    ## 2. 초기 설정② ####################################
    # i611 로봇 생성자
    rb = i611Robot()
    # 좌표계의 정의
    _BASE = Base()
    # 로봇과 연결 시작 초기화
    rb.open()
    # I/O 입출력 기능의 초기화
    IOinit(rb)

    ## 1. 교시 포인트 설정 ######################
    # 변수명은 임의로 지정이 가능하고 위에서 import해온 teachdata에서 꺼내쓰는것으로 보면 된다.
    # 사용하는 방식은 하기와 같다.

    ## 2. 동작 조건 설정 ########################
    # jnt_speed, lin_speed 설명, 한계값 jnt는 %개념이기에 100이 max, line_speed는 매뉴얼에 표기되어있음
    # acctime, dactime 설명_ 가속값, 감속값 설정한 속도까지 도달하기 위한 속도
    # 가감속 속도가 빠를수록 로봇의 택트는 감소하나 진동이 심할수 있음.
    # motionparam을 설정해서 로봇에 적용해야 원하는 퍼포먼스가 나온다.
    # 설정안할시 mcs 내부파일에 지정된 값으로 동작한다.n

    # set speed
    speed = 40
    motion = MotionParam(jnt_speed=speed, lin_speed=speed, pose_speed=speed, acctime=0.4, dacctime=0.4, overlap=20)

    # MotionParam 형으로 동작 조건 설정
    rb.motionparam(motion)  # 위에서 설정한 motionparameter를 로봇에 적용.

    rb.set_behavior(only_hook=True, servo_off=False, restore_position=True, no_pause=False)
    rb.enable_interrupt(0, True)  # 동작 중에 감속 정지 입력 시의 예외 발생을 활성화
    rb.enable_interrupt(1, True)  # 동작 중에 비상 정지 입력 시의 예외 발생을 활성화

    # taechdata를 꺼내쓰는것 외에 입력하는 방법도 있다.


    ## 3. 로봇 동작을 정의 ##############################
    # 작업 시작
    rb.home()  # 홈 위치로 이동
    initialPos = Joint(0,0,-90,0,-90,0)
    initialPos_2d = Position(-185, 590, 450, -90, 0,-180)
    initialPos_3d = Position(249, 550, 460, -180, 0,-180) # 초기위치로 이동


    test3 = Position(-86+104*math.cos(math.pi * 57 / 180), 534+104*math.sin(math.pi * 57 / 180), 250, 57, 0, -180)
    #rb.move(test2)

    dout(48, '000')  # Arm I/O 초기화
    #dout(48, '001')  # 그리퍼 close
    dout(48, '100')  # 그리퍼 open

    ##### 로봇 모션 테스트 구간 #####
    # testSline(rb)
    # testWline(rb)
    #testUline(rb)

    #jengaKnock(rb,-290,517,47)
    # move to initial 2d position to take picture
    rb.move(initialPos_2d)
    #rb.move(initialPos_3d)
    #jenga_upPlace(rb,-1,492,47)


    #jenga_upPlace(rb, -23, 651, -33)




    # jengaPlace(rb, -267, 618-43, 59)
    # jenga_sidePlace(rb, -267, 618, 59)#=> unreachable point
    #

    # test = Position(-100,418,544,90,45+90,0)=> 고침!
    # rb.move(test)
    #test = Position(-20)
    #rb.move(test)
    #jengaPlace(rb,-19,461-43,74)
    #jenga_sidePlace(rb, -19,461-43,74)

    #rb.move(initialPos)
    # time.sleep(0.2)
    # print('initialPos : ',initialPos.jnt2list())
    #
    # test_place_up = Position(-82, 528, 80, 58, -90, -180)
    # j = rb.Position2Joint(test_place_up)
    # print('test_place_up : ', j.jnt2list())
    # rb.move(test_place_up)


    #rb.move(initialPos_3d)
    #dout(48, '001')  # 그리퍼 close

    ## 1. server-client ####################################
    # 접속할 서버 주소입니다. 여기에서는 루프백(loopback) 인터페이스 주소 즉 localhost를 사용합니다. 
    HOST = '192.168.0.23'
    # 클라이언트 접속을 대기하는 포트 번호입니다.   
    PORT = 9989


    # 소켓 객체를 생성합니다. 
    # 주소 체계(address family)로 IPv4, 소켓 타입으로 TCP 사용합니다.  
    server_socket = socket(AF_INET, SOCK_STREAM)
    print('socket creat..')
    #server_socket = setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # bind 함수는 소켓을 특정 네트워크 인터페이스와 포트 번호에 연결하는데 사용됩니다.
    # HOST는 hostname, ip address, 빈 문자열 ""이 될 수 있습니다.
    # 빈 문자열이면 모든 네트워크 인터페이스로부터의 접속을 허용합니다. 
    # PORT는 1-65535 사이의 숫자를 사용할 수 있습니다.  
    server_socket.bind((HOST, PORT))
    print('bind end.. listen start..')
    # 서버가 클라이언트의 접속을 허용하도록 합니다. 
    server_socket.listen(1)
    print('listen end..')
    # accept 함수에서 대기하다가 클라이언트가 접속하면 새로운 소켓을 리턴합니다. 
    client_socket, addr = server_socket.accept()

    # 접속한 클라이언트의 주소입니다.
    print('Connected by', addr)

    ### 2d vision ###
    # 클라이언트로부터 15개의 x,y데이터와 각도를 받아옴
    data = client_socket.recv(1024)
    # coord15 리스트에 15개의 x,y데이터와 각도를 저장
    coord15 = []
    print('2d vision data : ',data.decode())


    # 수신받은 문자열을 출력합니다.
    for i in range(45):
        #print(data.split()[i].decode())
        coord15.append(float(data.split()[i].decode()))

    print('move start..')
    #time.sleep(0.1)
    dout(48, '100')  # 그리퍼 open
    rb.move(initialPos_3d)
    #time.sleep(0.1)
    cnt = 0

    debugMode = 0#1=디버그 모드

    try:
        for i in range(15):

            # 클라이언트가 보낸 메시지를 수신하기 위해 대기합니다.
            print('wait for data.. ')

            data = client_socket.recv(1024)
            # while len(data.split()) !=7:
            #     data = client_socket.recv(1024)

            positionData = getPositionData(data)
            label = getLabel(data)

            if debugMode == 1:
                ##### tact 카운트 시작 ########
                tact = time.time()  # tact start
                ##### 1줄씩 확인 #######
                pdb.set_trace()

            ####### jenga Pick
            jengaPick(rb, pick_x=positionData[0], pick_y=positionData[1], pick_z=positionData[2], pick_rz=positionData[3], pick_ry=positionData[4], pick_rx=positionData[5], label=label)


            ####### jenga Place
            #jengaPlace(rb, place_x=coord15[3 * cnt], place_y=coord15[3 * cnt + 1], recievedTheta=coord15[3 * cnt + 2])
            #
            if label == 'jenga':
                jengaPlace(rb, place_x=coord15[3*cnt], place_y=coord15[3*cnt+1], recievedTheta=coord15[3*cnt+2])
            elif label == 'jenga_side':
                jenga_sidePlace(rb, place_x=coord15[3 * cnt], place_y=coord15[3 * cnt + 1], recievedTheta=coord15[3 * cnt + 2])
            elif label == 'jenga_up':
                jenga_upPlace(rb, place_x=coord15[3 * cnt], place_y=coord15[3 * cnt + 1], recievedTheta=coord15[3 * cnt + 2])

            time.sleep(0.2)

            #rb.home()
            rb.move(initialPos)

            # move to 3d initial position to take picture
            rb.move(initialPos_3d)

            print("1 cycle 종료")

            if debugMode == 1:
                ####### tact 출력
                tact = time.time() - tact
                print
                "T/T : ", tact


            client_socket.sendall(b'finish')

            cnt+=1

        #knock jenga
        jengaKnock(rb, place_x=coord15[42], place_y=coord15[43], recievedTheta=coord15[44])
    
    except Robot_emo:
        print "Robot_emo"
        dout(48, '000')

    except KeyboardInterrupt:

        # "ctrl" + "c" 버튼 입력
        print "KeyboardInterrupt"
        dout(48, '000')
    finally:
        print "finally"
        dout(48, '000')
        rb.close()
    
    ## 4. 종료 ######################################
    # 로봇과의 연결을 종료
    rb.close()

    # 소켓을 닫습니다.
    client_socket.close()
    server_socket.close()

if __name__ == '__main__':
    main()
