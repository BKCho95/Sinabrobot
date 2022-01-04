# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified by Hyeun Jeong Min 2020

import atexit
import bisect
import multiprocessing as mp
from collections import deque
import cv2
import torch, time
import numpy as np
import math

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
import numpy as np

## communication message
MSG_IMG_REQ = 1     # from robot to img (camera start request)
MSG_IMG_DATA = 2    # from img to robot (recognition data)
MSG_IMG_NO_DATA = 12    # from img to robot (no recognition)
MSG_GRAB_FAIL = 3   # from robot to img (grabbing failure)
MSG_FAIL_RES = 4    # from img to robot (response)
PROCESS_QUIT = 100

UP_='UP'
DOWN_='DOWN'
LEFT_='LEFT'
RIGHT_='RIGHT'

def kMeans(X, K, maxIters=10, plot_progress=None):
    centroids = X[np.random.choice(np.arange(len(X)), K), :]
    for i in range(maxIters):
        # Cluster Assignment step
        C = np.array([np.argmin([np.dot(x_i - y_k, x_i - y_k) for y_k in centroids]) for x_i in X])
        # Move centroids step
        centroids = [X[C == k].mean(axis=0) for k in range(K)]
        c_num = [X[C == k] for k in range(K)]
        #print(len(centroids),centroids,c_num)
        if plot_progress != None: plot_progress(X, C, np.array(centroids))
    return np.array(centroids),c_num


class Visualization(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        #self.metadata = MetadataCatalog.get("data_").set(thing_classes=["chicken"])
        self.metadata = MetadataCatalog.get("data_").set(thing_classes=['jenga','jenga_side','jenga_up'])

        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)
            self.temp_var = []

    def send_no_data(self, m_in):
        send_msg = str(MSG_IMG_NO_DATA)
        m_in.send(send_msg)
        with open('a.txt', "a") as f:
            f.write(send_msg + ' ' + str(time.time()) + '\n')

    def run_on_image(self, image, depth_img, depth_scale, m_in, color_frame):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        twoDots = []
        vis_output = None
        predictions = self.predictor(image)
        #print(predictions)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

                ### inserted
                num_instances = len(instances)
                if num_instances > 0:
                    boxes1 = instances.pred_boxes.tensor.numpy() #predictions.pred_boxes.tensor.numpy() if predictions.has("pred_boxes") else None
                    scores1 = instances.scores #predictions.scores if predictions.has("scores") else None
                    max_idx = torch.nonzero((scores1==max(scores1)), as_tuple=False).item()
                    box = boxes1[max_idx]
                    ## revised
                    classes1 = instances.pred_classes
                    class1 = classes1.numpy()[max_idx]
                    score = round(scores1.numpy()[max_idx] * 100)
                    center_x = (box[0] + box[2]) / 2
                    center_y = (box[1] + box[3]) / 2
                    width = abs(box[2] - box[0])
                    height = abs(box[3] - box[1])
                    
                    ## revised
                    things_classes = ['jenga','jenga_side','jenga_up']
                    if class1 == 0: 
                        label = 'jenga'
                        print('******jenga')
                    elif class1 == 1: 
                        label = 'jenga_side'
                        print('******jenga_side')
                    elif class1 == 2: 
                        label = 'jenga_up'
                        print('******jenga_up')
                    
                    pp = image.shape
                    a1 = np.where(instances.pred_masks.numpy(), 1, 0)
                    aa1 = np.fromstring(a1[max_idx], dtype=int).reshape(pp[0], pp[1])
                    cordy, cordx = np.where(aa1 == 1)

                    ## PCA start (finding principal axis)
                    cordy = aa1.shape[0] - cordy-1
                    X = np.empty((len(cordx), 2))
                    X[:, 0], X[:, 1] = cordx, cordy
                    X_cen = X - X.mean(axis=0)
                    X_cov = np.dot(X_cen.T, X_cen) / (len(cordx) - 1)
                    w, v = np.linalg.eig(X_cov)

                    if (w[0] > w[1]):
                        vv = v[:, 0]
                    else:
                        vv = v[:, 1]

                    theta = np.arctan2(vv[1], vv[0])  #atan2(y,x)
                    #cv2.
                    # PCA end

                    # Center of Gravity (CoG) start
                    cog,C = kMeans(X, 1, maxIters=10, plot_progress=None)

                    cogx,cogy = np.round(cog[0][0]), vis_output.height-np.round(cog[0][1])
                    r_cent_x, r_cent_y = np.round(center_x), np.round(center_y)

                    diff_x, diff_y = cogx - r_cent_x, cogy-r_cent_y
                    if(abs(diff_x) > abs(diff_y)*.9): # LEFT/RIGHT
                        if(r_cent_x > cogx): direct = RIGHT_
                        else: direct = LEFT_
                    else: # UP/DOWN
                        if(r_cent_y > cogy): direct = DOWN_
                        else: direct = UP_

                    num_groups=3
                    cnd_cog,c_num = kMeans(X, num_groups, maxIters=20, plot_progress=None)
                    dx,dy,cnum=[],[],[]
                    for i in range(num_groups):
                        dx.append(cnd_cog[i][0])
                        dy.append(vis_output.height-cnd_cog[i][1])
                        cnum.append(len(c_num[i]))

                    minaid=cnum.index(min(cnum))

                    x=sorted(cnum)
                    for i in x:
                        minaid = cnum.index(i)
                        if(direct==DOWN_):
                            maxyid = dy.index(max(dy))
                            if (maxyid == minaid):
                                continue
                            else:
                                break
                        elif(direct==UP_):
                            maxyid = dy.index(min(dy))
                            if (maxyid == minaid):
                                continue
                            else:
                                break
                        elif(direct==LEFT_):
                            minxid = dx.index(min(dx))
                            if (minxid == minaid):
                                continue
                            else:
                                break
                        else:
                            minxid = dx.index(max(dx))
                            if (minxid == minaid):
                                continue
                            else:
                                break


                    cnd_cogx, cnd_cogy = np.round(cnd_cog[minaid][0]), vis_output.height - np.round(cnd_cog[minaid][1])

                    for i in range(-5,6):
                        for j in range(-5,6):
                            #print(i,j,cogy+i, cogx+j)
                            vis_output.img[int(cnd_cogy)+i,int(cnd_cogx)+j,0] = 0xFF
                            vis_output.img[int(cnd_cogy)+i,int(cnd_cogx)+j,1] = 0x00
                            vis_output.img[int(cnd_cogy)+i,int(cnd_cogx)+j,2] = 0x00

                    if (theta < 0): theta = theta+np.pi
                    ## PCA end

                    #depth start
                    temp_img = np.ones((pp[0],pp[1]))
                    #dist = depth_img.get_distance(int(cogx), int(cogy))
                    # alpha 52 = jennga size(7.2cm)
                    # alpha 52 = new jennga size(7.5cm)
                    
                    alpha = 26
                    dist1 = depth_img.get_distance(int(cogx+alpha*math.cos(theta)), int(cogy-alpha*math.sin(theta)))
                    dist2 = depth_img.get_distance(int(cogx-alpha*math.cos(theta)), int(cogy+alpha*math.sin(theta)))
                   
                    ## dist
                    sum = 0
                    cnt = 0
                    for i in range(3):
                        for j in range(3):
                            tmp = depth_img.get_distance(int(cogx-1+j), int(cogy-1+i))
                            if tmp != 0:
                                sum += tmp
                                cnt += 1

                    if cnt != 0:
                        dist = sum/cnt

                    else:
                        dist = 0

                    ## dist end
                    hop = 1
                    while dist1 == 0:
                        dist1 = depth_img.get_distance(int(cogx+alpha*math.cos(theta))+hop, int(cogy-alpha*math.sin(theta)+hop))
                        hop += 1
                    
                    hop = 1
                    while dist2 == 0:
                        dist2 = depth_img.get_distance(int(cogx+alpha*math.cos(theta))+hop, int(cogy-alpha*math.sin(theta)+hop))
                        hop += 1


                    twoDots.append(int(cogx+alpha*math.cos(theta)))
                    twoDots.append(int(cogy-alpha*math.sin(theta)))
                    twoDots.append(int(cogx-alpha*math.cos(theta)))
                    twoDots.append(int(cogy+alpha*math.sin(theta)))

                    tilt = np.arctan2(abs(dist1-dist2), 0.0375) 
                    sendTilt = np.rad2deg(tilt)
                    #depth end
                    print('real theta = ', theta)
                    sendTheta = (np.rad2deg(theta) +90)%180
                    
                    ## check around..
                    dist_around=[]
                    tmp = depth_img.get_distance(int(cogx-(8.3+5)*math.cos(theta)), int(cogy+alpha*math.sin(theta)))


                    send_msg = str(MSG_IMG_DATA) + ' ' \
                               + str(cogx) + ' ' + str(cogy) + ' ' + str(width) \
                               + ' ' + str(height) + ' ' + str(sendTheta) \
                               + ' ' + str(score) + ' ' + direct + ' ' + str(dist) \
                               + ' ' + str(cnd_cogx)+' '+str(cnd_cogy)+' '+str(dist1)+' '+str(dist2)  + ' ' + str(int(cogx-alpha)) +' '+ str(int(cogy-math.tan(theta))) \
                               + ' ' + str(int(cogx+alpha)) +' '+ str(int(cogy+math.tan(theta))) + ' ' + str(sendTilt) +' ' + label
                              
                    m_in.send(send_msg)

                    out = str(time.time())+ ' ' + str(num_instances)+' ' + str(cogx) + ' ' \
                          + str(cogy) + ' ' + str(r_cent_x) + ' ' + str(r_cent_y) + '\n'
                    with open('a.txt', "a") as f:
                        f.write(send_msg)
                        f.write('\n')
                        f.write(out)
                else:
                    self.send_no_data(m_in)

        return num_instances, vis_output, twoDots


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput a little bit when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5
