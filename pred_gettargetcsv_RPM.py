# 全資料夾轉成csv
from logging.config import valid_ident
import numpy as np
import pandas as pd
import math
import os
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def kfold(path):
    store_folder = os.path.dirname(os.path.realpath(__file__))
    csv_name = '175fps_5fold.csv'
    store_path = os.path.join(store_folder, csv_name)

    if os.path.exists(store_path):
        return pd.read_csv(store_path)
    else:
        path_list = os.listdir(path)
        data_df = []
        np.random.shuffle(path_list)
        start, end = 0, 0
        stride = math.ceil(len(path_list) / 5)
        # print(stride, len(path_list))
        for i in range(5):
            if end >= len(path_list):
                end = len(path_list) - 1
            else:
                end = end + stride
            data_df.append(path_list[start:end])
            start = end

        kfold = pd.DataFrame(data_df).T
        # print(df)
        # os.makedirs(store_path, exist_ok=True)
        # kfold.to_csv(store_folder + '/' + csv_name, index=False)
        kfold.to_csv(store_path, index=False)
        return pd.read_csv(store_path)

def five():
    ball_path = r'D:\Model_data\08camtemp\iphone3_ball_line'
    #kfold_csv = pd.read_csv(r"D:\Model_data\processed\csv\5fold.csv") # 如果有5fold.csv跑這個
    #kfold_csv = pd.read_csv(r'D:\Model_data\processed\csv\5fold.csv')
    #kfold_csv = kfold(ball_path)  # 如果沒有5fold.csv跑這個
    # kfold_csv = fourfold(ball_path)
    df_trackman = pd.read_csv(r'D:\Model_data\processed\csv\43881_20210924-TianMUStadium-1_unverified.csv')

    df_trackman1 = pd.read_csv(
        r'D:\Model_data\202204data\20220426-TianMUStadium-1_unverified.csv')
    df_trackman2 = pd.read_csv(
        r'D:\Model_data\202204data\20220428-TianMUStadium-1_unverified.csv')
    df_trackman3 = pd.read_csv(
        r'D:\Model_data\202204data\20220429-TianMUStadium-1_unverified.csv')
    df_trackman4 = pd.read_csv(
        r'D:\Model_data\202204data\20220522-TianMUStadium-1_unverified.csv')

    save_resultcsv_path = r'D:\My_Files\zly_python_file\baseball\python\175fps\5fold'

    for m in range(2):
        for n in range(5):
            df = pd.DataFrame(columns=['first', 'second', 'third', 'fourth', 'fifth', 'spinrate'])
            save_first = []
            save_second = []
            save_third = []
            save_fourth = []
            save_fifth = []
            spinrate_list = []
            Norm_spinrate_list = []
            minus_Norm_spinrate_list = []

            temp_list = kfold_csv[str(n)].tolist()

            test_folder_list = [x for x in temp_list if pd.isnull(x) == False and x != 'nan']
            all_folder_list = os.listdir(ball_path)

            for i in test_folder_list:
                print(i)
                all_folder_list.remove(i)
            train_folder_list = all_folder_list

            if (m == 0):
                folder_list = train_folder_list
                resultcsv_path = save_resultcsv_path + '\\' + "fivefold_trainset" + str(n) + ".csv"
            else:
                folder_list = test_folder_list
                resultcsv_path = save_resultcsv_path + '\\' + "fivefold_testset" + str(n) + ".csv"

            for p in tqdm(folder_list):
                # p = 2022xxxx_cam_5_XXX
                video_num = int(p.split("_")[-1])

                date = int(p.split("_")[0])
                if date == 20220426:
                    temp_df = df_trackman1.get(['VIDEOID', 'SpinRate'])
                elif date == 20220428:
                    temp_df = df_trackman2.get(['VIDEOID', 'SpinRate'])
                elif date == 20220429:
                    temp_df = df_trackman3.get(['VIDEOID', 'SpinRate'])
                elif date == 20220522:
                    temp_df = df_trackman4.get(['VIDEOID', 'SpinRate'])
                #
                # print(temp_df.loc[temp_df.VIDEOID == video_num].shape)
                # print(p)
                # print(temp_df)
                RPM = temp_df.loc[temp_df.VIDEOID == video_num].values[0][1]

                img_name_list = os.listdir(ball_path + '\\' + p)
                video_path = ball_path + '\\' + p
                img_name_list.sort(key=lambda x: int(x.split('.')[0]))
                for i in range(len(img_name_list) - 5 + 1):
                    if ((int(img_name_list[i].split('.')[0]) + 4) == int(img_name_list[i + 4].split('.')[0])):
                        save_first.append(video_path + '\\' + img_name_list[i])
                        save_second.append(video_path + '\\' + img_name_list[i + 1])
                        save_third.append(video_path + '\\' + img_name_list[i + 2])
                        save_fourth.append(video_path + '\\' + img_name_list[i + 3])
                        save_fifth.append(video_path + '\\' + img_name_list[i + 4])
                        spinrate_list.append(RPM)

            s_min = 800
            s_max = 3000
            for i in spinrate_list:
                norm = (i - s_min) / (s_max - s_min)
                Norm_spinrate_list.append(norm)
                minus_Norm_spinrate_list.append(1 - norm)

            df['first'] = save_first
            df['second'] = save_second
            df['third'] = save_third
            df['fourth'] = save_fourth
            df['fifth'] = save_fifth
            df['spinrate'] = spinrate_list
            df['Norm_spinrate'] = Norm_spinrate_list
            df["Norm_spinrate_minus"] = minus_Norm_spinrate_list

            df.to_csv(resultcsv_path)

#畫訓練集和測試集 且取每間隔相同數量
def calculate(folder_list, m):
    rpms = []
    for dirName in folder_list:
        rpm = int(dirName.split("_")[0])
        rpms.append(rpm)
    # 長方圖範圍1000到3000 間格為100
    bins_list = np.arange(1000, 3100, 100)
    n, bins, patches = plt.hist(rpms, bins=bins_list, edgecolor='black')  # 繪製直方圖
    plt.ylabel('Num')
    plt.xlabel('RPM')
    plt.xticks(bins_list, rotation='90', fontsize=8)  # 設定 X 軸標籤
    # plt.show()

    # 0是trainset 1是testset
    # trainset 每間隔取2600個
    if m == 0:
        for i in range(len(n)):
            n[i] = int(n[i] - 2600)
    # testset 每間隔取650個
    else:
        for i in range(len(n)):
            n[i] = int(n[i] - 650)

    for i in range(len(bins)-1):
        for dirName in folder_list:
            rpm = int(dirName.split("_")[0])
            rpm = int(rpm / 100) * 100
            if (rpm == bins[i]) and (n[i] > 0):

                folder_list.remove(dirName)
                n[i] -= 1

    rpms = []
    for dirName in folder_list:
        rpm = int(dirName.split("_")[0])
        rpms.append(rpm)
    n, bins, patches = plt.hist(rpms, bins=bins_list, edgecolor='black')

    # for p in patches:
    #     print(p)
    plt.ylabel('Num')
    plt.xlabel('RPM')
    plt.xticks(bins_list, rotation='90', fontsize=8)  # 設定 X 軸標籤
    # plt.show()
    #返回取相同數量後的list
    return folder_list

def fivefold_175fps():
    ball_path = './file/uploded_video_ball_line/08042022_113148.mov'
    # 之後要修改為上傳文件名的路徑
    #kfold_csv = pd.read_csv(r"D:\Model_data\processed\csv\5fold.csv") # 如果有5fold.csv跑這個
    kfold_csv = kfold(ball_path)  # 如果沒有5fold.csv跑這個

    save_resultcsv_path = './file/csv'

    for m in range(2):
        for n in range(5):
            df = pd.DataFrame(columns=['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', "nine" , "ten" ,  'spinrate'])
            save_1 = []
            save_2 = []
            save_3 = []
            save_4 = []
            save_5 = []
            save_6 = []
            save_7 = []
            save_8 = []
            save_9 = []
            save_10 = []
            spinrate_list = []
            Norm_spinrate_list = []
            minus_Norm_spinrate_list = []

            temp_list = kfold_csv[str(n)].tolist()

            test_folder_list = [x for x in temp_list if pd.isnull(x) == False and x != 'nan']
            all_folder_list = os.listdir(ball_path)

            for i in test_folder_list:
                all_folder_list.remove(i)
            train_folder_list = all_folder_list

            if (m == 0):
                folder_list = train_folder_list
                resultcsv_path = save_resultcsv_path + '\\' + "175fps_fivefold_trainset" + str(n) + ".csv"
            else:
                folder_list = test_folder_list
                resultcsv_path = save_resultcsv_path + '\\' + "175fps_fivefold_testset" + str(n) + ".csv"
            folder_list = calculate(folder_list, m)
            for p in tqdm(folder_list):
                # p = 2022xxxx_cam_5_XXX
                RPM = int(p.split("_")[0])

                img_name_list = os.listdir(ball_path + '\\' + p)
                video_path = ball_path + '\\' + p
                img_name_list.sort(key=lambda x: int(x.split('.')[0]))
                for i in range(len(img_name_list) - 10 + 1):
                    if ((int(img_name_list[i].split('.')[0]) + 9) == int(img_name_list[i + 9].split('.')[0])):
                        save_1.append(video_path + '\\' + img_name_list[i])
                        save_2.append(video_path + '\\' + img_name_list[i + 1])
                        save_3.append(video_path + '\\' + img_name_list[i + 2])
                        save_4.append(video_path + '\\' + img_name_list[i + 3])
                        save_5.append(video_path + '\\' + img_name_list[i + 4])
                        save_6.append(video_path + '\\' + img_name_list[i + 5])
                        save_7.append(video_path + '\\' + img_name_list[i + 6])
                        save_8.append(video_path + '\\' + img_name_list[i + 7])
                        save_9.append(video_path + '\\' + img_name_list[i + 8])
                        save_10.append(video_path + '\\' + img_name_list[i + 9])
                        spinrate_list.append(RPM)
            s_min = 900
            s_max = 3100
            for i in spinrate_list:
                norm = (i - s_min) / (s_max - s_min)
                Norm_spinrate_list.append(norm)
                minus_Norm_spinrate_list.append(1 - norm)

            df['first'] = save_1
            df['second'] = save_2
            df['third'] = save_3
            df['fourth'] = save_4
            df['fifth'] = save_5
            df['sixth'] = save_6
            df['seventh'] = save_7
            df['eighth'] = save_8
            df['nine'] = save_9
            df['ten'] = save_10
            df['spinrate'] = spinrate_list
            df['Norm_spinrate'] = Norm_spinrate_list
            df["Norm_spinrate_minus"] = minus_Norm_spinrate_list

            df.to_csv(resultcsv_path)

def getcsv(lineball_path):
    # ball_path = r'D:\My_Files\zly_python_file\baseball\python\server\file\uploded_video_ball_line'
    save_resultcsv_path = './file/csv/origin.csv'
    folder_path = lineball_path
    # folder_list = os.listdir(ball_path)
    # folder_list所有資料夾的list
    df = pd.DataFrame(columns=['first', 'second', 'third', 'fourth', 'fifth',  'spinrate','Norm_spinrate','Norm_spinrate_minus'])
    save_1 = []
    save_2 = []
    save_3 = []
    save_4 = []
    save_5 = []
    spinrate_list = []
    Norm_spinrate_list = []
    minus_Norm_spinrate_list = []
    
    # for p in tqdm(folder_list):
    #     # p = 2022xxxx_cam_5_XXX

    #     #
    #     # print(temp_df.loc[temp_df.VIDEOID == video_num].shape)
    #     # print(p)
    #     # print(temp_df)
    img_name_list = os.listdir(folder_path)
    video_path = folder_path
    #     video_path = ball_path + '\\' + p
    #     img_name_list.sort(key=lambda x: int(x.split('.')[0]))

    for i in range(len(img_name_list) - 5 + 1):
        if ((int(img_name_list[i].split('.')[0]) + 4) == int(img_name_list[i + 4].split('.')[0])):
            save_1.append(video_path + '\\' + img_name_list[i])
            save_2.append(video_path + '\\' + img_name_list[i + 1])
            save_3.append(video_path + '\\' + img_name_list[i + 2])
            save_4.append(video_path + '\\' + img_name_list[i + 3])
            save_5.append(video_path + '\\' + img_name_list[i + 4])

    df['first'] = save_1
    df['second'] = save_2
    df['third'] = save_3
    df['fourth'] = save_4
    df['fifth'] = save_5
    # df['spinrate'] = spinrate_list
    # df['Norm_spinrate'] = Norm_spinrate_list
    # df["Norm_spinrate_minus"] = minus_Norm_spinrate_list


    df.to_csv(save_resultcsv_path)
if __name__ == '__main__':
    # kfold(r'D:\My_Files\zly_python_file\baseball\test_pictures\2022_all_ball_line')
    # fivefold_175fps()
    # split()
    getcsv()