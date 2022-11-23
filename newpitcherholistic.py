import cv2
import mediapipe as mp
import numpy as np
import cProfile, pstats, io
from pstats import SortKey
import time
import threading
pr = cProfile.Profile()

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

mp_holistic = mp.solutions.holistic
IMAGE_FILES = []
# vedio
# cap = cv2.VideoCapture(r'C:\Users\samel\Downloads\drive-download-20220627T032831Z-001\cam_45024576_1_PX_V6.avi')
cap = cv2.VideoCapture('D:\\My_Files\\zly_python_file\\baseball\\python\\flaskserver\\test_file_src\\output_20221109165844.mov')

frame_index = 0
frame_count = 0 # frame_index / interval
videoFPS = 60

if cap.isOpened():
    success = True
else:
    print('openerror!')
    success = False

results = None
with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1) as holistic:
    pr.enable()
    
    seconds = time.time()
    # while cap.isOpened():
    while success:
        success, image = cap.read()
        # if not success:
        #     print("Ignoring empty camera frame")
        #     break
        
        # image.flags.writeable = False
        # image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        # print("Frame count", frame_index)
        if frame_index%1==0 and frame_index>50 and frame_index<300:
            results = holistic.process(image)
        elif frame_index==0:
            results = holistic.process(image)
        if frame_index>50 and frame_index<300:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2)
            )
        frame_index += 1
        IMAGE_FILES.append(image)
        # cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
        # cv2.imshow('MediaPipe Holistic', image)
        # if cv2.waitKey(5) & 0xFF == 27:
        #     break

    
    # 子執行緒的工作函數
    def job():
        frameSize = (1920, 1080)
        out = cv2.VideoWriter('D:\\My_Files\\zly_python_file\\baseball\\python\\flaskserver\\test_file_src\\ex_ballspeed.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 60, frameSize)
        for img in IMAGE_FILES:
            out.write(img)
        out.release()
    # 建立一個子執行緒
    t = threading.Thread(target = job)
    # 執行該子執行緒
    t.start()

    # frameSize = (1920, 1080)
    # out = cv2.VideoWriter('D:\\My_Files\\zly_python_file\\baseball\\python\\flaskserver\\test_file_src\\output_video.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 60, frameSize)
    # for img in IMAGE_FILES:
    #     out.write(img)
    # out.release()

    now = time.time()
    print("Time", now-seconds)
    pr.disable()
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())
cap.release()