import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
assert torch.__version__.startswith("1.9.1") #1.5



#from google.colab import drive
#drive.mount('./MyDrive')

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, random, cv2
#from google.colab import cv2_imshow
#from cv2 import *

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# labelme에서 만든 이미지 labeling 에서 정보를 가져와서 추출하는 코드
def dataset_func(dir):
  import os, json
  import numpy as np
  from detectron2.structures import BoxMode
  
  dataset_dict = []
  afile = os.listdir(dir)
  for jsonf in afile:
      print(jsonf)
      t = jsonf.find('.json')
      if (t>=0):
        print('filename:', jsonf)
      else: continue
      
      jsonf=dir+'/'+jsonf
      print(jsonf)
      with open(jsonf, encoding="utf8", errors='ignore') as f:
        imgs_anns = json.load(f)
        print(imgs_anns)
        
      fname = os.path.join(dir, imgs_anns['imagePath']) 
      print(fname)
      
      record={}
      record['file_name']=fname
      record['image_id']=0
      record['height']=imgs_anns['imageHeight']
      record['width']=imgs_anns['imageWidth']
      
      objs=[]
      for v in imgs_anns['shapes']:
        poly = v['points']
        poly = [p for x in poly for p in x]

######## 클래스 추가 #########
      if True:
        if(v['label']=='jenga'):
          cid = 0
        elif(v['label']=='jenga90'):
          cid = 1
        elif(v['label']=='jenga_up'):
          cid = 2
        else: cid = -1 #No category
    
      x1,y1 = np.min(poly[0::2]),np.min(poly[1::2])
      xe,ye = np.max(poly[0::2]),np.max(poly[1::2])
      obj = {
        'bbox': [x1,y1,xe,ye],
        'bbox_mode': BoxMode.XYXY_ABS,
        'segmentation': [poly],
        'category_id':cid
      }
      objs.append(obj)

      record['annotations']=objs
      dataset_dict.append(record)

  print(dataset_dict)
  return(dataset_dict)

for d in ["train", "val"]:
    DatasetCatalog.register("mdata_" + d, lambda d=d: dataset_func("/home/cobot/Downloads/" + d))
    MetadataCatalog.get("mdata_" + d).set(thing_classes=['jenga','jenga90','jenga_up'])
my_metadata = MetadataCatalog.get("mdata_train")

import random

dataset_dicts = dataset_func("/home/cobot/Downloads/train")
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=my_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    #cv2.imshow(out.get_image()[:, :, ::-1])

from detectron2.engine import DefaultTrainer

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
#### 데이터셋 이름 확인 
cfg.DATASETS.TRAIN = ("mdata_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 500    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
# 본인의 데이터 분류 클래스 개수 입력
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # only has one class (mount).

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()
