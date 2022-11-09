import os
from function import *
from cutBall import cutball
from pred_gettargetcsv_RPM import getcsv
from pred_RPM_pred_ip import pred

video_name = 'gen_frames_test.mp4'
video_path = './test_file_src/'+ video_name

lineball_path = cutball(video_path)
# getcsv(lineball_path)
# pred_spinrate = pred()
# print('pred_spinrate',pred_spinrate)