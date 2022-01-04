# -*- coding: ISO-8859-1 -*-

import os, time, tqdm
import cv2
import pyrealsense2 as rs
import numpy as np

from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger

from visualize import Visualization

import math

# constants
WINDOW_NAME = "Object Segmentation"

VAR_LAYER_CNT = 101
VAR_NUM_CLASSES = 3 #1
VAR_RES_DIR = './result'
VAR_OUTPUT_DIR = './output'
#VAR_OUTPUT_DIR = '../realsense#6/output'

## communication message
MSG_IMG_REQ = 1     # from robot to img (camera start request)
MSG_IMG_DATA = 2    # from img to robot (recognition data)
MSG_GRAB_FAIL = 3   # from robot to img (grabbing failure)
MSG_FAIL_RES = 4    # from img to robot (response)

from detectron2 import model_zoo

###
from PIL import Image
from PIL import ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import cv2

# ¸¶¿ì½º ÄÝ¹é ÇÔ¼ö: ¿¬¼ÓÀûÀÎ ¿øÀ» ±×¸®±â À§ÇÑ ÄÝ¹é ÇÔ¼ö
def DrawConnectedCircle(event, x, y, flags, param):
    global drawing
    
    # ¸¶¿ì½º ¿ÞÂÊ ¹öÆ°ÀÌ ´­¸®¸é µå·ÎÀ®À» ½ÃÀÛÇÔ
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        cv2.circle(img,(x,y),2,(0,0,255),-1)
        print(x, y) # ÁÂÇ¥ Á¤º¸ Ãâ·Â
    
    # ¸¶¿ì½º°¡ ¿ÞÂÊ ¹öÆ°À¸·Î ´­¸° »óÅÂ¿¡¼­ ¸¶¿ì½º Æ÷ÀÎÆ®¸¦ ¿òÁ÷ÀÌ¸é 
    # ¿òÁ÷ÀÎ ÀÚÃë¸¦ µû¶ó¼­ ¸¶¿ì½ºÀÇ Á¡µéÀÌ ±×·ÁÁü
    #elif event == cv2.EVENT_MOUSEMOVE:
        #if drawing == True:
            #cv2.circle(img,(x,y),2,(0,0,255),-1)
            #print(x, y)
    
    # ¸¶¿ì½º ¿ÞÂÊ ¹öÆ°À» ¶¼¸é µå·ÎÀ®À» Á¾·áÇÔ
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

def get15dots():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)


    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        #depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert images to numpy arrays
        #depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)

        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            break

    
   
    #cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
    # Filename
    # path = './sample/'
    imageName1 = str(time.strftime("%Y_%m_%d_%H_%M_%S")) +  '_Color.jpg'
  
    cv2.imwrite('./sample/'+ imageName1, color_image) 

    key = cv2.waitKey(1)
    # Press esc or 'q' to close the image window

    pipeline.stop()

    

    img = cv2.imread('./sample/'+imageName1)
    
    
    # x,y,w,h	= cv2.selectROI('img', img, False)
    # print('x,y,w,h = ',x,y,w,h)
    
    ################### automatically
    x=36
    y=50
    w=523
    h=352
    ###################

    if w and h:
        roi = img[y:y+h, x:x+w]
        cv2.imshow('cropped', roi)  
        cv2.moveWindow('cropped', 0, 0) 
        #cv2.imwrite('./cropped2.jpg', roi)   
        
        #img2 = img.copy()
        #a=420;b=297;c=420;d=297
        #a=150;b=500;c=350;dq=1300
        #a=y;b=y;c=w;d=h
        roi = img[y:y+h, x:x+w]
        roi2 = img[y:y+h, x:x+w].copy()
        roi3 = img[y:y+h, x:x+w].copy()
        roi4 = img[y:y+h, x:x+w].copy()


        
        imgray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        ret, imthres = cv2.threshold(imgray, 150, 255, cv2.THRESH_BINARY_INV) #  140,255.5

        
        contour, hierarchy = cv2.findContours(imthres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        #contour2, hierarchy = cv2.findContours(imthres, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
        print('contour length: %d'% (len(contour)))

        
        #cv2.drawContours(img, contour, -1, (0,255,0), 4)
        
        #cv2.drawContours(img2, contour2, -1, (0,255,0), 4)

        
        clist=[]
        dot_num_max=0
        dot_num_max_i = 0
        for i in range(len(contour)):
            cnt = 0
            for j in contour[i]:
                #cv2.circle(img, tuple(j[0]), 1, (255,0,0), -1)
                #print(tuple(j[0])[0])
                cnt += 1
            if dot_num_max < cnt:
                dot_num_max = cnt
                dot_num_max_i = i
            clist.append(cnt)

        print(dot_num_max, dot_num_max_i)

        step = (int)(clist[dot_num_max_i]/29)
        print('step:', step)

        dot = 0
        color = (255, 255, 0)

        
        dot_list=[]
        
        dot_list_15 =[]
        
        cnt = 0
        for j in contour[dot_num_max_i]:

            cnt += 1
            if cnt == step:
                cnt = 0
                cv2.circle(roi, tuple(j[0]), 5, color, cv2.FILLED)
                dot_list.append(j[0])
                dot+=1
                cv2.putText(roi, str(dot), tuple(j[0]),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)

        #dot_list_15
        def list_dot15_append(start_dot):
            j = start_dot-1
            dot_list_15.clear()
            dot_list_15.append(dot_list[j])
            for i in range(14):
                k = j + 1 + i
                if k > 28:
                    x = int((dot_list[j-1 - i][0] + dot_list[k-29][0]) / 2)
                    y = int((dot_list[j-1 - i][1] + dot_list[k-29][1]) / 2)

                else:
                    x = int((dot_list[j - 1 - i][0] + dot_list[k][0]) / 2)
                    y = int((dot_list[j - 1 - i][1] + dot_list[k][1]) / 2)

                a = []
                a.append(x)
                a.append(y)
                dot_list_15.append(a)
                
        
        cv2.imshow('BINARY', imthres)
        cv2.imshow('CONTOUR', roi)
        cv2.waitKey(0)
        input_startNum = int(input('input start num ; '))
        
        ###########################################
        list_dot15_append(input_startNum)
        ###########################################

        print(dot_list_15)

        

        cv2.imshow('DOT-15', roi2)

        for i in range(15):
            print(dot_list_15[i])

        
        gradient_normal_angle = []
        gradient_dot_angle =[]

        
        for i in range(14):#0~13
            dx = dot_list_15[i+1][0]-dot_list_15[i][0]
            dy = dot_list_15[i+1][1]-dot_list_15[i][1]
            if dy == 0:
                dy = 0.01
            angle = np.degrees(np.arctan(dx / dy * -1))
            gradient_normal_angle.append(angle)
        gradient_normal_angle.append(gradient_normal_angle[13])

       
        gradient_dot_angle.append(gradient_normal_angle[0])
        for i in range(13):
            angle = (gradient_normal_angle[i]+gradient_normal_angle[i+1])/2
            if gradient_normal_angle[i]*gradient_normal_angle[i+1]<0:
                angle = angle-90
            gradient_dot_angle.append(angle)
        gradient_dot_angle.append(gradient_normal_angle[13])

        gradient_dot_angle[13] = 0
        gradient_dot_angle[1] = 0

        z=20
        for i in range(15):
            #find 
            if (gradient_dot_angle[i] == 0 and i>=1 and i<=13):
                gradient_dot_angle[i] = (gradient_dot_angle[i-1] + gradient_dot_angle[i+1])/2

        # normal data angle result GOOD..
        
        for j in range(len(dot_list_15)):
            cv2.circle(roi3, tuple(dot_list_15[j]), 5, color, cv2.FILLED)
            cv2.putText(roi3, str(j+1), tuple(dot_list_15[j]), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
            dx=int(z*np.cos(np.deg2rad(gradient_normal_angle[j])))
            dy=int(z*np.sin(np.deg2rad(gradient_normal_angle[j])))
            print('dx,dy, gradient_normal_angle[j] = ',dx,dy, gradient_normal_angle[j])
            roi3 = cv2.line(roi3, (dot_list_15[j][0]-dx, dot_list_15[j][1]-dy), (dot_list_15[j][0]+dx, dot_list_15[j][1]+dy), (0, 0, 255), 3, cv2.LINE_AA)

        
        # for j in range(len(dot_list_15)):
        #     cv2.circle(roi4, tuple(dot_list_15[j]), 5, color, cv2.FILLED)
        #     cv2.putText(roi4, str(tuple(dot_list_15[j])), tuple(dot_list_15[j]), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        #     dx=int(z*np.cos(np.deg2rad(gradient_dot_angle[j])))
        #     dy=int(z*np.sin(np.deg2rad(gradient_dot_angle[j])))
        #     print(dx,dy)
        #     roi4 = cv2.line(roi4, (dot_list_15[j][0]-dx, dot_list_15[j][1]-dy), (dot_list_15[j][0]+dx, dot_list_15[j][1]+dy), (0, 0, 255), 3, cv2.LINE_AA)

        print(gradient_normal_angle)
        print(gradient_dot_angle)

    
    
    #### 2d vision calibration ####
    # robot x
    X=np.mat([[0.02091
    ],
            [-0.28666
    ],
            [0.01360
    ],
            [-0.29452
    ],
            [-0.13644]])

    # robot y
    Y=np.mat([[0.44118
    ],
            [0.42930
    ],
            [0.65470
    ],      
            [0.64234
    ],
            [0.54121]])
    

    # image
    Prime=np.mat([[ 94, 102, 1],
                [514, 99, 1],
                [96, 391, 1],
                [516, 389, 1],
                [305, 244, 1]])

    dx_matrix = Prime.I*X
    dy_matrix = Prime.I*Y
    
    dot15Data = ''

    for i in range(15):
        image_x = dot_list_15[i][0] + x
        image_y = dot_list_15[i][1] + y

        # unit : mm
        dx = (dx_matrix[0, 0]*image_x + dx_matrix[1, 0]*image_y + dx_matrix[2, 0])*1000
        dy = (dy_matrix[0, 0]*image_x + dy_matrix[1, 0]*image_y + dy_matrix[2, 0])*1000

        if (gradient_dot_angle[i] == 0 and i>=1 and i<=13):
            gradient_dot_angle[i] = (gradient_dot_angle[i-1] + gradient_dot_angle[i+1])/2
        tmp = str(dx) +' ' + str(dy) + ' ' +str(gradient_dot_angle[i])
        dot15Data += tmp + ' '
    
    
  
    

    cv2.imshow('NORMAL', roi3)
    #v2.imshow('FINAL', roi4)
   
    
    

    return dot15Data



def pil_draw_point(image, point):
    x, y = point
    draw = ImageDraw.Draw(image)
    radius = 2
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(0, 0, 255))

    return image
###

def get_coord_distance(color_intrin, depth_to_color_extrin, x, y, dist, depth_scale):
    depth_point = rs.rs2_deproject_pixel_to_point(color_intrin,[x,y],dist*depth_scale)
    [x3d,y3d,z3d] = rs.rs2_transform_point_to_point(depth_to_color_extrin,depth_point)
    [x3d,y3d,z3d] = [x3d*1000000,y3d*1000000,z3d*1000000]
    return [x3d, y3d, z3d]

def setup_cfg(path):
    cfg = get_cfg()
    cfg.MODEL.DEVICE='cpu'
    if (VAR_LAYER_CNT == 50):
        cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
    else:
        cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml'))
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = VAR_NUM_CLASSES  # only has one class (chicken)
    cfg.MODEL.WEIGHTS = os.path.join(path, "model_final.pth")
    ################################### threshold ######################################
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05  # set a custom testing threshold
    cfg.freeze()
    return cfg

def test_main(pipe):
    setup_logger(name="fvcore")

    m_out, m_in = pipe

    r_path = VAR_RES_DIR
    m_path = VAR_OUTPUT_DIR
    cfg = setup_cfg(m_path)
    os.makedirs(r_path, exist_ok=True)

    s_time = time.time()
    with open("a.txt", "a") as f:
        aa='\nStart '+str(s_time)+'\n'
        f.write(aa)

    print('meta: ', cfg.DATASETS.TEST[0], cfg.DATASETS)
    vis = Visualization(cfg)

    pipeline = rs.pipeline()
    config = rs.config()
    dev = rs.device()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    align_to = rs.stream.color
    align = rs.align(align_to)

    idx=0
    while True:
        data = m_out.recv()
        print('Cam: received data: ', data)

        if(data == MSG_IMG_REQ):
           

            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            aligned_depth_frames = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not aligned_depth_frames or not color_frame: continue
            
            color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
            depth_intrin = aligned_depth_frames.profile.as_video_stream_profile().intrinsics
            depth_to_color_extrin = aligned_depth_frames.profile.get_extrinsics_to(color_frame.profile)
            depth_frame = aligned_depth_frames.get_data()

            depth_img = np.asanyarray(depth_frame)
            color_img = np.asanyarray(color_frame.get_data())

            img = color_img
            twoDots = []
            num_instances, v_output, twoDots = vis.run_on_image(img,aligned_depth_frames, depth_scale, m_out, color_frame)

            if num_instances == -1:
                continue
            #time.sleep(1)

            fname = 'orgimg_' + str(idx) + '.jpg'
            out_1 = os.path.join(r_path, fname)
            if len(twoDots) == 4:
                cv2.circle(img, (twoDots[0],twoDots[1]), 5, (0,0,255))
                cv2.circle(img, (twoDots[2],twoDots[3]), 5, (0,255,0))
            cv2.imwrite(out_1, img)
            

            fname = 'orgdepth_' + str(idx) + '.jpg'
            
            out_2 = os.path.join(r_path, fname)
            cv2.imwrite(out_2, depth_img)

            fname = 'img_' + str(idx) + '.jpg'
            out_filename = os.path.join(r_path, fname)
            
            idx += 1
            v_output.save(out_filename)

        else:
            time.sleep(1)

    rs.release()
    cv2.destroyAllWindows()

from multiprocessing import Pipe, Process
from task1 import call_task1


def task(pipe):
    call_task1(pipe, dot15Data)

dot15Data = ''

if __name__ == '__main__':

    ######## 2d vision part ########
    dot15Data = get15dots()
    print('*****sendData =', dot15Data)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    ######## 3d vision part ########
    m_out, m_in = Pipe()
    p1 = Process(target=test_main, args=((m_out,m_in),))
    p2 = Process(target=task, args=((m_out,m_in),))
    p1.start()
    p2.start()


