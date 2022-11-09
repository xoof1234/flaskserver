import cv2
import os
import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf


def create_folder(path):
    # imgPath = './ball/' + date + '_' + foldName
    if not os.path.isdir(path):
        os.mkdir(path)
    else:
        pass
    return path


def cutframe_iphone(video_name):
    ball_frames = []
    video_frame = []
    loss_frame = []
    ball_frame_names = []
    history = 500
    varThreshold = 180
    bShadowDetection = True
    #背景切割器
    mog = cv2.createBackgroundSubtractorMOG2(history,varThreshold,bShadowDetection)
    #
    es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cap = cv2.VideoCapture(video_name)

    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    #開始幀數
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_count = 0
    img_count = 0
    while (1):

        ret, frame = cap.read()

        if ret:
            try:
                #video_frame.append(frame)

                #背景切割器
                fgmask = mog.apply(frame)
                th = cv2.threshold(fgmask, 244, 255, cv2.THRESH_BINARY)[1]
                # cv2.imshow('th', th)
                
                # 開運算
                opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, es, iterations=1)
                # cv2.imshow('opening', opening)

                cont, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

                for c in cont:
                    ROI_xleft = 800
                    ROI_xright = 1600
                    ROI_ytop = 400
                    ROI_ydown = 800
                    cv2.rectangle(frame,(ROI_xleft,ROI_ytop),(ROI_xright,ROI_ydown),(0,255,0),2)
                    #面積
                    if (cv2.contourArea(c) < 1800 and (cv2.contourArea(c) > 270)):

                        (x, y, w, h) = cv2.boundingRect(c)
                        #if (abs(w - h) < 10 and (w < 40 and h < 40) and y < 500):
                        #if (abs(w - h) < 10 and (w < 35 and h < 35) and (y < 450) and (y > 250) and (x > 325)):
                        #if (abs(w - h) < 10 and (w < 40 and h < 40) and (y > 200) and (y < 600) and (x > 700) and (x < 1000)):
                        if (abs(w - h) < 10  and (w < 60 and h < 60) and (y > ROI_ytop) and (y < ROI_ydown) and (x > ROI_xleft) and (x < ROI_xright)):
                            #print(cv2.contourArea(c))
                            ROI = frame[(y - 5):(y + h + 5), (x - 5):(x + w + 5)]

                            ROI_hsv = cv2.cvtColor(ROI, cv2.COLOR_BGR2HSV)
                            v_temp = round((np.mean(ROI_hsv[:, :, 2]) / 52.87), 2)
                            ROI_hsv[:, :, 2] = ROI_hsv[:, :, 2] / v_temp
                            ROI_hsv = cv2.cvtColor(ROI_hsv, cv2.COLOR_HSV2BGR)
                            #亮度調整
                            img = modify_lightness_saturation(
                                ROI_hsv)

                            img = cv2.resize(img, (48, 48))

                            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

                            ball_frames.append(img)
                            ball_frame_names.append(frame_count)

                frame_count += 1
                # video_frame.append(frame)




            except cv2.error as e:
                print(e)
                continue
        else:
            break

    cap.release()

    # return np.array(video_frame),np.array(ball_frames),ball_frame_names
    return np.array(ball_frames),ball_frame_names



def cutframe_cam(video_name):
    ball_frames = []
    video_frame = []
    loss_frame = []
    ball_frame_names = []
    history = 500
    varThreshold = 180
    bShadowDetection = True
    mog = cv2.createBackgroundSubtractorMOG2(history,varThreshold,bShadowDetection)
    #
    es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cap = cv2.VideoCapture(video_name)

    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 150)
    frame_count = 0

    while (1):

        ret, frame = cap.read()

        if ret:
            try:
                fgmask = mog.apply(frame)



                th = cv2.threshold(fgmask, 244, 255, cv2.THRESH_BINARY)[1]
                # cv2.imshow('th', th)
                
                # 開運算
                opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, es, iterations=1)
                # cv2.imshow('opening', opening)

                cont, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
                # print(type(cont))

                for c in cont:
                    #(x, y, w, h) = cv2.boundingRect(c)
                    ROI_xleft = 350
                    ROI_xright = 600
                    ROI_ytop = 50
                    ROI_ydown = 350
                    cv2.rectangle(frame,(ROI_xleft,ROI_ytop),(ROI_xright,ROI_ydown),(0,255,0),2)


                    if (cv2.contourArea(c) < 450 and (cv2.contourArea(c) > 150)):

                        (x, y, w, h) = cv2.boundingRect(c)
                        #if (abs(w - h) < 10 and (w < 40 and h < 40) and y < 500):
                        #if (abs(w - h) < 10 and (w < 35 and h < 35) and (y < 450) and (y > 250) and (x > 325)):
                        #if (abs(w - h) < 10 and (w < 40 and h < 40) and (y > 200) and (y < 600) and (x > 700) and (x < 1000)):
                        if (abs(w - h) < 10  and (w < 30 and h < 30) and (y > ROI_ytop) and (y < ROI_ydown) and (x > ROI_xleft) and (x < ROI_xright)):
                            #print(cv2.contourArea(c))
                            ROI = frame[(y - 5):(y + h + 5), (x - 5):(x + w + 5)]

                            # ROI_hsv = cv2.cvtColor(ROI, cv2.COLOR_BGR2HSV)
                            # v_temp = round((np.mean(ROI_hsv[:, :, 2]) / 52.87), 2)
                            # ROI_hsv[:, :, 2] = ROI_hsv[:, :, 2] / v_temp

                            # ROI_hsv = cv2.cvtColor(ROI_hsv, cv2.COLOR_HSV2BGR)

                            # img = modify_lightness_saturation(
                            #     ROI)
                            img = cv2.resize(ROI, (48, 48))

                            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

                            ball_frames.append(img)
                            ball_frame_names.append(frame_count)

                            # cv2.imshow('ball', img)

                            # img_count += 1
                            # fileName = 'D:\\Model_data\\08camtemp\\img\\{}.png'.format(img_count)
                            # cv2.imwrite(fileName, img)
                            
                frame_count += 1            
                # cv2.imshow('frame', frame)
                # cv2.waitKey(50)   

                video_frame.append(frame)

            except cv2.error as e:
                print(e)
                continue

        else:
            break
    cap.release()

    return np.array(video_frame),np.array(ball_frames),ball_frame_names





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


def cutframe_th(video_name):
    ball_frames = []
    video_frame = []
    loss_frame = []
    history = 500
    varThreshold = 180
    bShadowDetection = True
    mog = cv2.createBackgroundSubtractorMOG2(history,varThreshold,bShadowDetection)
    #
    es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cap = cv2.VideoCapture(video_name)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
    count = 0
    while (count <90):

        ret, frame = cap.read()

        if ret:
            try:
                now_temp = len(ball_frames)
                video_frame.append(frame)

                fgmask = mog.apply(frame)
                th = cv2.threshold(fgmask, 244, 255, cv2.THRESH_BINARY)[1]
                opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, es, iterations=1)

                cont, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

                for c in cont:
                    area = cv2.contourArea(c)
                    if (area > 100 and area < 300):
                        (x, y, w, h) = cv2.boundingRect(c)

                        if (abs(w - h) < 10 and (w < 40 and h < 40) and y < 400):  #abs(w - h) < 10 and (w < 30 and h < 30) and y < 200
                            tw = int(w * 0.33)
                            th = int(h * 0.33)
                            ROI = frame[(y - th):(y + h + th), (x - tw):(x + w + tw)]
                            # ROI = frame[(y - 5):(y + h + 5), (x - 5):(x + w + 5)]

                            ROI_hsv = cv2.cvtColor(ROI, cv2.COLOR_BGR2HSV)
                            v_temp = round((np.mean(ROI_hsv[:, :, 2]) / 52.87), 2)
                            ROI_hsv[:, :, 2] = ROI_hsv[:, :, 2] / v_temp

                            ROI_hsv = cv2.cvtColor(ROI_hsv, cv2.COLOR_HSV2BGR)

                            img = modify_lightness_saturation(
                                ROI_hsv)
                            img = cv2.resize(img, (48, 48))

                            cv2.rectangle(frame, (x, y), (x + w + w, y + h + h), (0, 255, 0), 2)

                            ball_frames.append(img)




                next_temp = len(ball_frames)
                if((next_temp - now_temp) == 0):
                    loss_frame.append(count)





            except cv2.error as e:
                break




        else:
            break
        if len(ball_frames) >= 45:
            break

        count += 1

    cap.release()

    return np.array(video_frame),np.array(ball_frames),loss_frame



if __name__ == '__main__':
    try:
        video_frames,ball_frames,loss_frames = cutframe_cam('D:\\Model_data\\08camtemp\\2022-08-04\\cam_13224297_79_PX_VX.avi')
    except Exception as e:
        print(e)    
    # for i in range(len(ball_frames)):
    #     cv2.imshow("3", video_frames[i])
    #     # cv2.imshow("2",ball_frames[i])
    #     cv2.waitKey(100)  
