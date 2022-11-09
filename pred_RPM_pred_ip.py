import matplotlib.pyplot as plt
import random
import tensorflow as tf
import pandas as pd
import cv2
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

def root_mean_squared_error(y_true, y_pred):
    msle = tf.keras.losses.MeanSquaredLogarithmicError()
    return K.sqrt(msle(y_true, y_pred))

def pred():
    print("pred start...")
    # model = load_model(r'C:\Users\Ricky\PycharmProjects\server\h5\10_175fps_2022_09_07_train0_rmse.h5' ,custom_objects={'root_mean_squared_error': root_mean_squared_error})
    model = load_model('./model/spinrate__240FPS.ckpt' ,custom_objects={'root_mean_squared_error': root_mean_squared_error})

    #model.load_weights(r'C:\Users\maxchen\Desktop\Project\code\RPM_h5\RPM_resN_5fold.h5')
    #df = pd.read_csv(r'D:\Model_data\processed\csv\RPM.csv')
    model.summary()
    df = pd.read_csv('./file/csv/origin.csv')
    #print(df.iloc[0][1])

    test_predict = []
    test_y_true = []
    s_min = 900
    s_max = 3100

    for i in range(int(len(df))):
        inputs = []
        for j in range(1,11):
            path = df.iloc[i][j]
            img = cv2.imread(path)
            resImg = cv2.resize(img, (48, 48))
            resImg = np.expand_dims(resImg, axis=0)
            resImg = resImg / 255.0
            inputs.append(resImg)

        RPM = df.iloc[i][11]
        test_y_true.append(RPM)

        predict = model.predict(inputs)

        test_predict.append(predict)
        #print(predict)
        #print(type(predict)) #numpy.ndarray
    test_predict_temp = []
    for i in range(len(test_predict)):
        pred = test_predict[i][0][0] * (s_max - s_min) + s_min
        test_predict_temp.append(pred)
        #print(test_predict[i][0][0])

    df["pred"] = test_predict_temp
    df.to_csv('./file/csv/result.csv')
    res = df["pred"].mean()
    print("pred_result:",res)

    return res

# from sklearn.metrics import r2_score

# mse = mean_squared_error(test_y_true,test_predict_temp)
# mae = mean_absolute_error(test_y_true,test_predict_temp)
# rmse = np.sqrt(mean_squared_error(test_y_true,test_predict_temp))
# r_square = r2_score(test_y_true,test_predict_temp)

# print(f'mse = {mse}')
# print(f'mae = {mae}')
# print(f'rmse = {rmse}')
# print(f'r_square = {r_square}')

# # ny= np.asarray(test_y_true)
# # ny_bar = np.asarray(test_predict_temp)
# # r =  1 - sum((ny - ny_bar)**2) / sum((ny - np.mean(ny))**2)
# # r

# plt.figure(figsize=(10, 10))
# for i in range(len(test_predict_temp)):
#     #temp = random.randint(0, 1000)test_y_true[i]
#     plt.scatter(test_predict_temp[i], test_y_true[i], c='red')
#     #plt.scatter(test_y_true[temp], test_predict_temp[temp], c='red')

# test_y_min = np.round(int(min(test_y_true)), -2) - 250
# test_y_max = np.round(int(max(test_y_true)), -2) + 250
# plt.xticks(range(test_y_min, test_y_max,50),rotation = '90')
# plt.yticks(range(test_y_min, test_y_max,50))

# plt.xlim(900, 3100)
# # 設定y軸的取值範圍為：-1到3
# #max(test_predict_temp)+300
# plt.ylim(900, 3100)

# # 生成x軸上的資料:從-3到3，總共有50個點
# x = np.linspace(0, max(test_predict_temp)+300, 20)
# # 定義一個線性方程
# y1 = x
# # 繪製紅色的線寬為1虛線的線條
# plt.plot(x, y1, color='blue', linewidth=2.0, linestyle='--')
# plt.title(f'rmse: {rmse:.3f}  r^2: {r_square:.3f}', fontsize=16)
# plt.ylabel('true for spinrate')
# plt.xlabel('predict for spinrate')
# plt.savefig('2022_8_29_175fps_test0.png')
# plt.show()
