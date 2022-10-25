# pip install pybase64
import pybase64

# f = open('t.txt')
# contents = f.read()
# videoData = pybase64.b64decode(contents)
# filename = "123.ap4"
# with open(filename,"wb") as f:
#     f.write(videoData)

with open("fastball1.mp4", "rb") as videoFile:
    text = pybase64.b64encode(videoFile.read())
    print(text)
    file = open("fastball1.txt", "wb")
    file.write(text)
    file.close()

    # fh = open("video.mp4", "wb")
    # fh.write(pybase64.b64decode(text))
    # fh.close()