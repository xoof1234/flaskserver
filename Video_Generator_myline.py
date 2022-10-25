import cv2
import numpy as np
import os

from tensorflow.keras.utils import to_categorical


from sklearn.utils import shuffle
import  tensorflow as tf
from tensorflow.keras.utils import Sequence


def modify_lightness_saturation(img):
    origin_img = img

    # 圖像歸一化，且轉換為浮點型
    fImg = img.astype(np.float32)
    fImg = fImg / 255.0

    # 顏色空間轉換 BGR -> HLS
    hlsImg = cv2.cvtColor(fImg, cv2.COLOR_BGR2HLS)
    hlsCopy = np.copy(hlsImg)

    lightness = 130  # lightness 調整為  "1 +/- 幾 %"
    saturation = 0  # saturation 調整為 "1 +/- 幾 %"

    # 亮度調整
    hlsCopy[:, :, 1] = (1 + lightness / 100.0) * hlsCopy[:, :, 1]
    hlsCopy[:, :, 1][hlsCopy[:, :, 1] > 1] = 1  # 應該要介於 0~1，計算出來超過1 = 1

    # 飽和度調整
    hlsCopy[:, :, 2] = (1 + saturation / 100.0) * hlsCopy[:, :, 2]
    hlsCopy[:, :, 2][hlsCopy[:, :, 2] > 1] = 1  # 應該要介於 0~1，計算出來超過1 = 1

    # 顏色空間反轉換 HLS -> BGR
    result_img = cv2.cvtColor(hlsCopy, cv2.COLOR_HLS2RGB)
    result_img = ((result_img * 255).astype(np.uint8))

    return result_img

def mormallize(img):
    new_image = np.zeros(img.shape, img.dtype)
    alpha = 1.8
    beta = 50
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            for c in range(img.shape[2]):
                new_image[y, x, c] = np.clip(alpha * img[y, x, c] + beta, 0, 255)
    return  new_image
class DataGenerator(Sequence):

    def __init__(self, dataFrame, batch_size, Size):
        self.dataFrame = dataFrame
        self.batch_size = batch_size
        self.size = (Size, Size)

    def __len__(self):
        return int(np.ceil(len(self.dataFrame.target)) / float(self.batch_size))

    def __getitem__(self, idx):
        origin_path_batch = self.dataFrame.origin.values[idx * int(self.batch_size): (idx + 1) * int(self.batch_size)]
        target_path_batch = self.dataFrame.target.values[idx * int(self.batch_size): (idx + 1) * int(self.batch_size)]
        item = self.__generator(origin_path_batch, target_path_batch)
        return item

    def on_epoch_end(self): #??
        self.dataFrame = self.dataFrame

    def __generator(self, origin_path_batch, target_path_batch):
        concat_images = list()
        labels = list()
        for Image in target_path_batch: # 0-32路徑

            label = cv2.imread(Image)
            #label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
            label = cv2.resize(label, self.size)
            label = label / 255.0
            labels.append(label)

        for Image in origin_path_batch:
            real_ball = cv2.imread(Image)
            real_ball = cv2.resize(real_ball, self.size)
            real_ball = real_ball / 255.0
            concat_images.append(real_ball)


            # if('C:/Users/xoof/Desktop/Baseball-20211023T060056Z-001/Baseball/' in Image):

            #     temp = Image.split('/')
            #     folderName = temp[-2].replace('(line)', '')
            #     ImgName = temp[-1]

            #     path = './bright_ball/'
            #     trueImage = cv2.imread(path + folderName + '/' + ImgName)

            #     trueImage = cv2.cvtColor(trueImage, cv2.COLOR_BGR2RGB)
            #     trueImage = cv2.resize(trueImage, self.size)
            #     # trueImage_bright = mormallize(trueImage)
            #     # gray = cv2.cvtColor(trueImage_bright, cv2.COLOR_BGR2GRAY)
            #     # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            #     # canny = cv2.Canny(blurred, 30, 150)
            #     # canny = np.expand_dims(canny,axis=-1)

            #     # trueImage_bright= np.concatenate((trueImage_bright,canny),axis=-1)
            #     result_img = trueImage / 255.0
            # elif('D:/code/baseball/123/201-300-l/' in Image):
            #     temp = Image.split('/')

            #     trueImage = cv2.imread('D:/code/baseball/123/201-300/' + temp[-1])
            #     trueImage = cv2.cvtColor(trueImage, cv2.COLOR_BGR2RGB)
            #     trueImage = cv2.resize(trueImage, self.size)
            #     # trueImage_bright = mormallize(trueImage)
            #     # gray = cv2.cvtColor(trueImage_bright, cv2.COLOR_BGR2GRAY)
            #     # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            #     # canny = cv2.Canny(blurred, 30, 150)
            #     # canny = np.expand_dims(canny, axis=-1)

            #     # trueImage_bright = np.concatenate((trueImage_bright, canny), axis=-1)
            #     result_img = trueImage / 255.0
            # elif ('D:/code/baseball/123/101-200-l/' in Image):
            #     temp = Image.split('/')
            #     trueImage = cv2.imread('D:/code/baseball/123/101-200/' + temp[-1])
            #     trueImage = cv2.cvtColor(trueImage, cv2.COLOR_BGR2RGB)
            #     trueImage = cv2.resize(trueImage, self.size)
            #     # trueImage_bright = mormallize(trueImage)
            #     # gray = cv2.cvtColor(trueImage_bright, cv2.COLOR_BGR2GRAY)
            #     # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            #     # canny = cv2.Canny(blurred, 30, 150)
            #     # canny = np.expand_dims(canny, axis=-1)

            #     # trueImage_bright = np.concatenate((trueImage_bright, canny), axis=-1)
            #     result_img = trueImage / 255.0
            # elif ('D:/code/baseball/123/1-100-l/' in Image):
            #     temp = Image.split('/')

            #     trueImage = cv2.imread('D:/code/baseball/123/1-100/' + temp[-1])
            #     trueImage = cv2.cvtColor(trueImage, cv2.COLOR_BGR2RGB)
            #     trueImage = cv2.resize(trueImage, self.size)
            #     # trueImage_bright = mormallize(trueImage)
            #     # gray = cv2.cvtColor(trueImage_bright, cv2.COLOR_BGR2GRAY)
            #     # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            #     # canny = cv2.Canny(blurred, 30, 150)
            #     # canny = np.expand_dims(canny, axis=-1)
            #     #
            #     # trueImage_bright = np.concatenate((trueImage_bright, canny), axis=-1)
            #     result_img = trueImage / 255.0
            # elif ('C:/Users/xoof/Desktop/123/line/' in Image):
            #     temp = Image.split('/')

            #     trueImage = cv2.imread('C:/Users/xoof/Desktop/123/72ppi/' + temp[-1])
            #     trueImage = cv2.resize(trueImage, self.size)
            #     result_img = trueImage / 255.0
            # else:
            #     temp = Image.split('/')
            #     trueImage = cv2.imread('D:/code/baseball/bright_ball/cam_5_497/' + temp[-1])
            #     trueImage = cv2.cvtColor(trueImage, cv2.COLOR_BGR2RGB)
            #     trueImage = cv2.resize(trueImage, self.size)
                # trueImage_bright = mormallize(trueImage)
                # gray = cv2.cvtColor(trueImage_bright, cv2.COLOR_BGR2GRAY)
                # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                # canny = cv2.Canny(blurred, 30, 150)
                # canny = np.expand_dims(canny, axis=-1)

                # trueImage_bright = np.concatenate((trueImage_bright, canny), axis=-1)
                # result_img = trueImage / 255.0

            # concat_images.append(result_img)
        # print(np.asarray(concat_images).shape)

        return (np.asarray(concat_images), np.asarray(labels))

class DataGenerator_myRPM(Sequence):

    def __init__(self, dataFrame, batch_size, Size):
        self.dataFrame = dataFrame
        self.batch_size = batch_size
        self.size = (Size, Size)

    def __len__(self):
        return int(np.ceil(len(self.dataFrame) / float(self.batch_size)))

        #return int(np.ceil(len(self.dataFrame.target)) / float(self.batch_size))
    def __getitem__(self, idx):
        first_path_batch = self.dataFrame.get('first').iloc[idx * int(self.batch_size): (idx + 1) * int(self.batch_size)]
        second_path_batch = self.dataFrame.get('second').iloc[idx * int(self.batch_size): (idx + 1) * int(self.batch_size)]
        third_path_batch = self.dataFrame.get('third').iloc[idx * int(self.batch_size): (idx + 1) * int(self.batch_size)]
        fourth_path_batch = self.dataFrame.get('fourth').iloc[idx * int(self.batch_size): (idx + 1) * int(self.batch_size)]
        fifth_path_batch = self.dataFrame.get('fifth').iloc[idx * int(self.batch_size): (idx + 1) * int(self.batch_size)]
        norm_spinrate_batch = self.dataFrame.get('Norm_spinrate').iloc[idx * int(self.batch_size): (idx + 1) * int(self.batch_size)]
        minus_norm_spinrate_batch = self.dataFrame.get('Norm_spinrate_minus').iloc[idx * int(self.batch_size): (idx + 1) * int(self.batch_size)]
        # norm_spinrate_batch.reset_index(inplace = True, drop=True)
        # minus_norm_spinrate_batch.reset_index(inplace = True, drop=True)
        # print(minus_norm_spinrate_batch.keys())
        item = self.__generator(first_path_batch, second_path_batch, third_path_batch, fourth_path_batch, fifth_path_batch, norm_spinrate_batch, minus_norm_spinrate_batch)
        return item

    def on_epoch_end(self): #??
        self.dataFrame = self.dataFrame

    def __generator(self, first_path_batch, second_path_batch, third_path_batch, fourth_path_batch, fifth_path_batch, norm_spinrate_batch, minus_norm_spinrate_batch):
        first_balls = list()
        second_balls = list()
        third_balls = list()
        fourth_balls = list()
        fifth_balls = list()
        labels = []
        #labels = [[0] * 2 for i in range(2)]

        for Image in first_path_batch: # 0-32路徑
            first_ball = cv2.imread(Image)
            # label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
            first_ball = cv2.resize(first_ball, self.size)
            first_ball = first_ball / 255.0
            first_balls.append(first_ball)

        for Image in second_path_batch: # 0-32路徑

            second_ball = cv2.imread(Image)
            # label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
            second_ball = cv2.resize(second_ball, self.size)
            second_ball = second_ball / 255.0
            second_balls.append(second_ball)

        for Image in third_path_batch: # 0-32路徑

            third_ball = cv2.imread(Image)
            # label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
            third_ball = cv2.resize(third_ball, self.size)
            third_ball = third_ball / 255.0
            third_balls.append(third_ball)

        for Image in fourth_path_batch: # 0-32路徑

            fourth_ball = cv2.imread(Image)
            # label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
            fourth_ball = cv2.resize(fourth_ball, self.size)
            fourth_ball = fourth_ball / 255.0
            fourth_balls.append(fourth_ball)

        for Image in fifth_path_batch: # 0-32路徑

            fifth_ball = cv2.imread(Image)
            # label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
            fifth_ball = cv2.resize(fifth_ball, self.size)
            fifth_ball = fifth_ball / 255.0
            fifth_balls.append(fifth_ball)
        # for i in norm_spinrate_batch:
        #     labels.append(i)
        for i, j in zip(norm_spinrate_batch, minus_norm_spinrate_batch):
            labels.append([i, j])

        # for i in range(len(norm_spinrate_batch)):
        #     labels.append([norm_spinrate_batch[i], minus_norm_spinrate_batch[i]])
        return ([np.asarray(first_balls),np.asarray(second_balls),np.asarray(third_balls),np.asarray(fourth_balls),np.asarray(fifth_balls)], np.asarray(labels))


class DataGenerator_RPM_175fps(Sequence):

    def __init__(self, dataFrame, batch_size, Size):
        self.dataFrame = dataFrame
        self.batch_size = batch_size
        self.size = (Size, Size)

    def __len__(self):
        return int(np.ceil(len(self.dataFrame) / float(self.batch_size)))

        # return int(np.ceil(len(self.dataFrame.target)) / float(self.batch_size))

    def __getitem__(self, idx):
        first_path_batch = self.dataFrame.get('first').iloc[
                           idx * int(self.batch_size): (idx + 1) * int(self.batch_size)]
        second_path_batch = self.dataFrame.get('second').iloc[
                            idx * int(self.batch_size): (idx + 1) * int(self.batch_size)]
        third_path_batch = self.dataFrame.get('third').iloc[
                           idx * int(self.batch_size): (idx + 1) * int(self.batch_size)]
        fourth_path_batch = self.dataFrame.get('fourth').iloc[
                            idx * int(self.batch_size): (idx + 1) * int(self.batch_size)]
        fifth_path_batch = self.dataFrame.get('fifth').iloc[
                           idx * int(self.batch_size): (idx + 1) * int(self.batch_size)]
        sixth_path_batch = self.dataFrame.get('sixth').iloc[
                           idx * int(self.batch_size): (idx + 1) * int(self.batch_size)]
        seventh_path_batch = self.dataFrame.get('seventh').iloc[
                             idx * int(self.batch_size): (idx + 1) * int(self.batch_size)]
        eighth_path_batch = self.dataFrame.get('eighth').iloc[
                            idx * int(self.batch_size): (idx + 1) * int(self.batch_size)]
        nine_path_batch = self.dataFrame.get('nine').iloc[
                            idx * int(self.batch_size): (idx + 1) * int(self.batch_size)]
        ten_path_batch = self.dataFrame.get('ten').iloc[
                            idx * int(self.batch_size): (idx + 1) * int(self.batch_size)]
        norm_spinrate_batch = self.dataFrame.get('Norm_spinrate').iloc[
                              idx * int(self.batch_size): (idx + 1) * int(self.batch_size)]
        minus_norm_spinrate_batch = self.dataFrame.get('Norm_spinrate_minus').iloc[
                                    idx * int(self.batch_size): (idx + 1) * int(self.batch_size)]
        # norm_spinrate_batch.reset_index(inplace = True, drop=True)
        # minus_norm_spinrate_batch.reset_index(inplace = True, drop=True)
        # print(minus_norm_spinrate_batch.keys())
        item = self.__generator(first_path_batch, second_path_batch, third_path_batch, fourth_path_batch,
                                fifth_path_batch, sixth_path_batch, seventh_path_batch, eighth_path_batch, nine_path_batch, ten_path_batch,
                                norm_spinrate_batch, minus_norm_spinrate_batch)
        return item

    def on_epoch_end(self):  # ??
        self.dataFrame = self.dataFrame

    def __generator(self, first_path_batch, second_path_batch, third_path_batch, fourth_path_batch,
                                fifth_path_batch, sixth_path_batch, seventh_path_batch, eighth_path_batch, nine_path_batch, ten_path_batch,
                                norm_spinrate_batch, minus_norm_spinrate_batch):
        first_balls = list()
        second_balls = list()
        third_balls = list()
        fourth_balls = list()
        fifth_balls = list()
        sixth_balls = list()
        seventh_balls = list()
        eighth_balls = list()
        nine_balls = list()
        ten_balls = list()
        labels = []
        # labels = [[0] * 2 for i in range(2)]

        for Image in first_path_batch:  # 0-32路徑
            first_ball = cv2.imread(Image)
            # label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
            first_ball = cv2.resize(first_ball, self.size)
            first_ball = first_ball / 255.0
            first_balls.append(first_ball)

        for Image in second_path_batch:  # 0-32路徑

            second_ball = cv2.imread(Image)
            # label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
            second_ball = cv2.resize(second_ball, self.size)
            second_ball = second_ball / 255.0
            second_balls.append(second_ball)

        for Image in third_path_batch:  # 0-32路徑

            third_ball = cv2.imread(Image)
            # label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
            third_ball = cv2.resize(third_ball, self.size)
            third_ball = third_ball / 255.0
            third_balls.append(third_ball)

        for Image in fourth_path_batch:  # 0-32路徑

            fourth_ball = cv2.imread(Image)
            # label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
            fourth_ball = cv2.resize(fourth_ball, self.size)
            fourth_ball = fourth_ball / 255.0
            fourth_balls.append(fourth_ball)

        for Image in fifth_path_batch:  # 0-32路徑

            fifth_ball = cv2.imread(Image)
            # label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
            fifth_ball = cv2.resize(fifth_ball, self.size)
            fifth_ball = fifth_ball / 255.0
            fifth_balls.append(fifth_ball)

        for Image in sixth_path_batch:  # 0-32路徑

            sixth_ball = cv2.imread(Image)
            # label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
            sixth_ball = cv2.resize(sixth_ball, self.size)
            sixth_ball = sixth_ball / 255.0
            sixth_balls.append(sixth_ball)
        for Image in seventh_path_batch:  # 0-32路徑

            seventh_ball = cv2.imread(Image)
            # label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
            seventh_ball = cv2.resize(seventh_ball, self.size)
            seventh_ball = seventh_ball / 255.0
            seventh_balls.append(seventh_ball)
        for Image in eighth_path_batch:  # 0-32路徑

            eighth_ball = cv2.imread(Image)
            # label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
            eighth_ball = cv2.resize(eighth_ball, self.size)
            eighth_ball = eighth_ball / 255.0
            eighth_balls.append(eighth_ball)
        for Image in nine_path_batch:  # 0-32路徑

            nine_ball = cv2.imread(Image)
            # label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
            nine_ball = cv2.resize(nine_ball, self.size)
            nine_ball = nine_ball / 255.0
            nine_balls.append(nine_ball)
        for Image in ten_path_batch:  # 0-32路徑

            ten_ball = cv2.imread(Image)
            # label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
            ten_ball = cv2.resize(ten_ball, self.size)
            ten_ball = ten_ball / 255.0
            ten_balls.append(ten_ball)

        for i in norm_spinrate_batch:
            labels.append(i)
        # for i, j in zip(norm_spinrate_batch, minus_norm_spinrate_batch):
        #     labels.append([i, j])

        # for i in range(len(norm_spinrate_batch)):
        #     labels.append([norm_spinrate_batch[i], minus_norm_spinrate_batch[i]])
        return ([np.asarray(first_balls), np.asarray(second_balls), np.asarray(third_balls), np.asarray(fourth_balls),
                 np.asarray(fifth_balls), np.asarray(sixth_balls), np.asarray(seventh_balls), np.asarray(eighth_balls) , np.asarray(nine_balls), np.asarray(ten_balls)],
                np.asarray(labels))


class DataGenerator_valid(Sequence):

    def __init__(self, path, dataFrame, batch_size, Size):
        self.path = path
        self.dataFrame = dataFrame.sample(frac=1, replace=False)
        self.batch_size = batch_size
        self.size = (Size, Size)

    def __len__(self):
        return int(np.ceil(len(self.dataFrame.path)) / float(self.batch_size))

    def __getitem__(self, idx):

        path_batch = self.dataFrame.path.values[idx * int(self.batch_size): (idx + 1) * int(self.batch_size)]

        item = self.__generator(path_batch)
        return item

    def on_epoch_end(self):
        self.path = self.path

    def __generator(self, path_batch):
        concat_images = list()
        # print(path_batch)
        for ImageFolder in path_batch:
            wrong = False
            List = list()  # Save clip folder images name
            images = list()
            Count = 0
            FolderPath = self.path + ImageFolder
            for i in os.listdir(FolderPath):
                List.append(self.path + ImageFolder + '/' + i)
            List = sorted(List)
            Count = 0
            for j in range(5):

                if (Count == 0):
                    try:
                        Img1 = cv2.imread(List[j])
                        resImg1 = cv2.resize(Img1, self.size)
                        Img = resImg1
                    except cv2.error as e:
                        print(ImageFolder)
                        wrong = True
                else:
                    try:
                        Img1 = cv2.imread(List[j])
                        resImg1 = cv2.resize(Img1, self.size)
                        Img = np.concatenate((Img, resImg1), -1)
                    except cv2.error as e:
                        print(ImageFolder)
                        wrong = True
                Count += 1
            Img = Img / 255.0
            if (wrong == False):
                concat_images.append(Img)

        return (np.asarray(concat_images))



class DataGenerator_RPM_valid(Sequence):

    def __init__(self, path, dataFrame, batch_size, Size):
        self.path = path
        self.dataFrame = dataFrame.sample(frac=1, replace=False)
        self.batch_size = batch_size
        self.size = (Size, Size)

    def __len__(self):
        return int(np.ceil(len(self.dataFrame.path)) / float(self.batch_size))

    def __getitem__(self, idx):

        path_batch = self.dataFrame.path.values[idx * int(self.batch_size): (idx + 1) * int(self.batch_size)]
        item = self.__generator(path_batch)
        return item

    def on_epoch_end(self):
        self.path = self.path

    def __generator(self, path_batch):
        concat_images = list()

        for ImageFolder in path_batch:
            wrong = False
            List = list()  # Save clip folder images name

            FolderPath = self.path + ImageFolder
            for i in os.listdir(FolderPath):
                List.append(self.path + ImageFolder + '/' + i)
            List = sorted(List)
            Count = 0
            for j in range(5):

                if (Count == 0):
                    try:
                        Img1 = cv2.imread(List[j])
                        resImg1 = cv2.resize(Img1, self.size)
                        Img = resImg1
                    except cv2.error as e:
                        print(ImageFolder)
                        wrong = True
                else:
                    try:
                        Img1 = cv2.imread(List[j])
                        resImg1 = cv2.resize(Img1, self.size)
                        Img = np.concatenate((Img, resImg1), -1)
                    except cv2.error as e:
                        print(ImageFolder)
                        wrong = True
                Count += 1
            Img = Img / 255.0
            if (wrong == False):
                concat_images.append(Img)

        return (np.asarray(concat_images))

class DataGenerator_RPM(Sequence):

    def __init__(self, path, dataFrame, batch_size, Size):
        self.path = path
        self.dataFrame = dataFrame.sample(frac=1, replace=False)
        self.batch_size = batch_size
        self.size = (Size, Size)

    def __len__(self):
        return int(np.ceil(len(self.dataFrame.path)) / float(self.batch_size))

    def __getitem__(self, idx):

        path_batch = self.dataFrame.path.values[idx * int(self.batch_size): (idx + 1) * int(self.batch_size)]
        item = self.__generator(path_batch)
        return item

    def on_epoch_end(self):
        self.path = self.path

    def __generator(self, path_batch):
        concat_images = list()
        RPMS = list()
        for ImageFolder in path_batch:
            wrong = False
            List = list()  # Save clip folder images name
            temp = ImageFolder.split('_')
            RPM = int(temp[0])
            FolderPath = self.path + ImageFolder
            for i in os.listdir(FolderPath):
                List.append(self.path + ImageFolder + '/' + i)
            List = sorted(List)
            Count = 0
            for j in range(5):

                if (Count == 0):
                    try:
                        Img1 = cv2.imread(List[j])
                        resImg1 = cv2.resize(Img1, self.size)
                        Img = resImg1
                    except cv2.error as e:
                        print(ImageFolder)
                        wrong = True
                else:
                    try:
                        Img1 = cv2.imread(List[j])
                        resImg1 = cv2.resize(Img1, self.size)
                        Img = np.concatenate((Img, resImg1), -1)
                    except cv2.error as e:
                        print(ImageFolder)
                        wrong = True
                Count += 1
            Img = Img / 255.0
            if (wrong == False):
                concat_images.append(Img)
                RPMS.append(RPM)
        return (np.asarray(concat_images)), (np.asarray(RPMS))



