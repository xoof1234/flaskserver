from PIL import Image
 
# 将图片转化为RGB
def make_regalur_image(img, size=(64, 64)):
    gray_image = img.resize(size).convert('RGB')
    return gray_image
 
# 计算直方图
def hist_similar(lh, rh):
    assert len(lh) == len(rh)
    hist = sum(1 - (0 if l == r else float(abs(l-r))/max(l,r))for l, r in zip(lh, rh))/len(lh)
    return hist
 
# 计算相似度
def calc_similar(li, ri):
    calc_sim = hist_similar(li.histogram(), ri.histogram())
    return calc_sim
 
if __name__ == '__main__':
    image1 = Image.open('./file/uploded_video_ball/trim/30.png')
    image1 = make_regalur_image(image1)
    image2 = Image.open('./file/uploded_video_ball/trim/31.png')
    image2 = make_regalur_image(image2)
    print("图片间的相似度为",calc_similar(image1, image2))
