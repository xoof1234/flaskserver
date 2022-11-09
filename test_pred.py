import os
from function import *
from cutBall import cutball
from pred_gettargetcsv_RPM import getcsv
from pred_RPM_pred_ip import pred

video_name = 'output_20221109163418.mov'
video_path = './test_file_src/'+ video_name

lineball_path = cutball(video_path)
lineball_path = './file/uploded_video_ball_line/output_20221109163418'
# getcsv(lineball_path)
# pred_spinrate = pred()
# print('pred_spinrate',pred_spinrate)