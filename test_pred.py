import os
from function import *
from cutBall import cutball
from pred_gettargetcsv_RPM import getcsv
from pred_RPM_pred_ip import pred
import time

video_name = 'output_20221109163418.mov'
video_path = './test_file_src/'+ video_name

time_start = time.time()

cutball(video_path)

time_end = time.time()
print('processing time', time_end - time_start, 's')

# lineball_path = './file/uploded_video_ball_line/output_20221109163418'
# getcsv(lineball_path)
# pred_spinrate = pred()
# print('pred_spinrate',pred_spinrate)

