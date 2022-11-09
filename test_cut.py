import cv2
 
def calculate(image1, image2):
    # 灰度直方图算法
    # 计算单通道的直方图的相似值
    hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
    # 计算直方图的重合度
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + \
                (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:
            degree = degree + 1
    degree = degree / len(hist1)
    return degree
 
if __name__ == '__main__':
    image1 = cv2.imread('./file/uploded_video_ball/output_ex/297.png',0)
    image2 = cv2.imread('./file/uploded_video_ball/output_ex/316.png',0)
    print("图片间的相似度为",calculate(image1, image2))
