import numpy as np
import argparse
#import imutils
import time
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from matplotlib.pyplot import imshow
import glob
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer)
from utils.torch_utils import select_device, load_classifier, time_synchronized

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import  ActivityRegularization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import glob

import cv2
import matplotlib.pyplot as plt








def load_model(weights,device):
    
    
    # Loading Character model
    device = str(device)
    device = select_device(device)
    model = attempt_load(weights, map_location=device)  # load FP32 model
    
    return model, device
    



def detect(model,
           output,
           immg,
           source= 'test/',
           save_img=True,
           view_img=True,
           save_txt=True,
           img_size=416,
           device='0',
           conf_thres=0.7,
           iou_thres=0.5,
           classes=None,
           agnostic_nms=True,
          augment=True):
    #print("#"*10)
    
    out, source,  view_img, save_txt, imgsz, conf_thres, iou_thres, classes, agnostic_nms = \
        output, source, view_img, save_txt, img_size, conf_thres, iou_thres, classes, agnostic_nms
    
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    #device = select_device(device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Set Dataloader
    
    save_img = True
        #dataset = LoadImages(source, img_size=imgsz)
        
    #img0 = immg[...,::-1]
    img0 = cv2.cvtColor(immg, cv2.COLOR_RGB2BGR)
    img = letterbox(img0, imgsz, 32)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    dataset = [['', img, img0, None]]

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    #print(names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        #print(img)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=augment)[0]
        
        #print(pred)
        #print("SECOOOOOMD")

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        t2 = time_synchronized()
            
        # Process detections
        pred_cord = []
        pred_cord_xy_wh = []
        for i, det in enumerate(pred):  # detections per image
            
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s
            save_path  = str(Path(out) / Path(p).name)
            txt_path = '' # str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()                
                
                # Print results
                for c in det[:, -1].detach().unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        pred_cord.append(torch.tensor(xyxy).view(1, 4).tolist())
                        pred_cord_xy_wh.append(xywh)
#                     if save_img or view_img:  # Add bbox to image
#                         xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist() 
#                         label = '%s' % (names[int(cls)])
#                         #print(xyxy[0])
#                         #plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
            # return predictions with orignal image
            return pred_cord, im0
    

    

def sort_pred(predictions):
    
    # create predictions_2 which is a list of list (go down a list) so it's easier to sort
    predictions_2 = []
    for p in predictions:
        predictions_2.append(p[0])
        
    # sort predictions_2 based on y1, this way we get the objects in the same line together
    predictions_2.sort(key = lambda predictions_2: predictions_2[1])
    
    #we can't always assume there will be 7 characters in a line, so we will get half the length of the list to divide the lines
    char_per_line = int(len(predictions_2)/2) 
    
    upper_line = predictions_2[0:char_per_line]
    lower_line = predictions_2[char_per_line:]
    
    # sort each line based on x1
    upper_line.sort(key = lambda upper_line: upper_line[0])
    lower_line.sort(key = lambda lower_line: lower_line[0])
    
    predictions_2 = upper_line + lower_line
    
    #return this to its original format [[['like],['this]]]
    predictions_joined = []
    for i in predictions_2:
        predictions_joined.append([i])

    return predictions_joined





def plot_pred(img, predictions):
    
    for pred in predictions:
        #x, y, w, h  = pred
        x_min, y_min, x_max, y_max  = pred[0]
        x_min, y_min, x_max, y_max = int(x_min), int(y_min) , int(x_max) , int(y_max)
        H, W, _ = img.shape

        plt.subplot(1,2,1)
        plt.imshow(img)
        plt.subplot(1,2,2)
        #img[y_min:y_max, x_min:x_max]
        plt.imshow(img[y_min:y_max, x_min:x_max])

        #plt.imshow(img[y:y+h, x:x+w])
        plt.show()
#         name = 'croped_plate/' + str(cnt) + '.jpg'
#         print(type(img[y_min:y_max, x_min:x_max]))
#         cv2.imwrite(name,img[y_min:y_max, x_min:x_max])
#         cnt += 1





def crop_pred(img, predictions):
    cropped_images = []
    for pred in predictions:

        x_min, y_min, x_max, y_max  = pred[0]
        x_min, y_min, x_max, y_max = int(x_min), int(y_min) , int(x_max) , int(y_max)
        #H, W, _ = img.shape
        cropped_images.append(img[y_min:y_max, x_min:x_max])
    
    return cropped_images
