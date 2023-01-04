import os
from function import *
from cutBall import cutball
from pred_gettargetcsv_RPM import getcsv
from pred_RPM_pred_ip import pred
from time import perf_counter
import pandas as pd

def get_dataframe(ball_to_line_img, ball_frame_names):
    print("size in func:",len(ball_to_line_img))
    print("type in func:",type(ball_to_line_img))
    print("ball_frame_names in func:",ball_frame_names)

    df = pd.DataFrame(columns=['first','second','third','fourth','fifth','spinrate','Norm_spinrate','Norm_spinrate_minus'])
    save_1 = []
    save_2 = []
    save_3 = []
    save_4 = []
    save_5 = []
    spinrate_list = []
    Norm_spinrate_list = []
    minus_Norm_spinrate_list = []

    img_name_list = ball_frame_names

    for i in range(len(img_name_list) - 5 + 1):
        if ((int(img_name_list[i]) + 4) == int(img_name_list[i + 4])):
            save_1.append(ball_to_line_img[i])
            save_2.append(ball_to_line_img[i + 1])
            save_3.append(ball_to_line_img[i + 2])
            save_4.append(ball_to_line_img[i + 3])
            save_5.append(ball_to_line_img[i + 4])

    df['first'] = save_1
    df['second'] = save_2
    df['third'] = save_3
    df['fourth'] = save_4
    df['fifth'] = save_5
    # df['spinrate'] = spinrate_list
    # df['Norm_spinrate'] = Norm_spinrate_list
    # df["Norm_spinrate_minus"] = minus_Norm_spinrate_list
    return df

video_name = '2022101529.mov'
video_path = '.\\file\\uploded_video\\'+ video_name

time_start = perf_counter()

ball_to_line_img,ball_frame_names = cutball(video_path)

time_end = perf_counter()
print('processing time', time_end - time_start, 's')

df = get_dataframe(ball_to_line_img, ball_frame_names)
# lineball_path = './file/uploded_video_ball_line/output_20221109163418'
# getcsv(lineball_path)
pred_spinrate = pred(df)
# print('pred_spinrate',pred_spinrate)



