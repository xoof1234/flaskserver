import cv2
import os
import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf

import argparse
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from numpy import random

from yoloV7_model.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
import warnings
def create_folder(path):
    # imgPath = './ball/' + date + '_' + foldName
    if not os.path.isdir(path):
        os.mkdir(path)
    else:
        pass
    return path

class ball_cutframe:
    def __init__(self,video_name) :
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
        parser.add_argument('--source', type=str, default= video_name, help='source')  # file/folder, 0 for webcam
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default='runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--no-trace', action='store_true', help='don`t trace model')


        self.args = parser.parse_args()
        self.weight = self.args.weight
        self.source = self.args.source
        self.img_size = self.args.img-size
        self.view_img = self.args.view_img
        self.save_txt = self.args.save_txt
        self.imgsz = self.args.imgsz
        self.trace =  not self.args.no_trace
        
        self.save_img = not self.args.nosave and not self.source.endswith('.txt')
        self.save_dir = Path(increment_path(Path(self.args.project) / self.args.name, exist_ok=self.args.exist_ok))
        (self.save_dir / 'labels' if self.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
        set_logging()
        self.device = select_device(self.args.device)
        self.half = self.device.type != 'cpu' 
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.imgsz, s=stride)  # check img_size
        if self.trace:
            self.model = TracedModel(self.model, self.device, self.args.img_size)

        if self.half:
            self.model.half()  # to FP16

        self.dataset = LoadImages(self.source, img_size=self.imgsz, stride=stride)
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        self.old_img_w = self.old_img_h = self.imgsz
        self.old_img_b = 1
        self.ball_extract_frame()

    def ball_extract_frame(self):
         for path, img, ori_img, vid_cap in self.dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if self.device.type != 'cpu' and (self.old_img_b != img.shape[0] or self.old_img_h != img.shape[2] or self.old_img_w != img.shape[3]):
                self.old_img_b = img.shape[0]
                self.old_img_h = img.shape[2]
                self.old_img_w = img.shape[3]
                for i in range(3):
                    self.model(img,augment=self.args.augment)[0]

            # Inference
            
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = self.model(img,augment=self.args.augment)[0]
            

            # Apply NMS
            pred = non_max_suppression(pred, self.args.conf_thres, self.args.iou_thres, classes=self.args.classes, agnostic=self.args.agnostic_nms)
            

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                p, s, im0, frame = path, '', ori_img, getattr(self.dataset, 'frame', 0)
                p = Path(p)  # to Path
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape).round()
                    det_list = det.cpu().numpy().tolist()
                    det_list.sort(key = lambda det_list: det_list[4],reverse=True)
                    for tmp_ in range(len(det_list)):
                        x1, y1, x2, y2, conf, classes = det_list[tmp_]
                        if(classes == 32.0):
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2),int(y2)
                            self.ball.append(cv2.resize(ori_img[y1:y2, x1:x2],(48,48)))
                            self.frame.append(ori_img)
        
    def extract_ball_return(self):
        return self.ball
    def frame_return(self):
        return self.frame





def modify_lightness_saturation(img):
    f_Img = img.astype(np.float32)
    f_Img = f_Img / 255.0

    hls_img = cv2.cvtColor(f_Img, cv2.COLOR_BGR2HLS)
    hls_copy = np.copy(hls_img)

    lightness = 130
    saturation = 0

    hls_copy[:, :, 1] = (1 + lightness / 100.0) * hls_copy[:, :, 1]
    hls_copy[:, :, 1][hls_copy[:, :, 1] > 1] = 1

    hls_copy[:, :, 2] = (1 + saturation / 100.0) * hls_copy[:, :, 2]
    hls_copy[:, :, 2][hls_copy[:, :, 2] > 1] = 1

    result_img = cv2.cvtColor(hls_copy, cv2.COLOR_HLS2RGB)
    result_img = ((result_img * 255).astype(np.uint8))


    return result_img


def root_mean_squared_error(y_true, y_pred):
    msle = tf.keras.losses.MeanSquaredLogarithmicError()
    return K.sqrt(msle(y_true, y_pred))




if __name__ == '__main__':
    try:
        video_frames,ball_frames,loss_frames = cutframe_cam('D:\\Model_data\\08camtemp\\2022-08-04\\cam_13224297_79_PX_VX.avi')
    except Exception as e:
        print(e)    
    # for i in range(len(ball_frames)):
    #     cv2.imshow("3", video_frames[i])
    #     # cv2.imshow("2",ball_frames[i])
    #     cv2.waitKey(100)  
