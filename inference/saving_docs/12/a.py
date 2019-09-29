import sys
sys.path.append('/home/xingduan/caffe_parallel/python')
import cv2
import numpy as np

def draw_rect(img, rect, color):
    # Use BlueViolet to save predict face
    # print "img.shape: ", img.shape
    # print "rect: ", rect
    rect = map(int, rect)
    x1 = rect[0]
    y1 = rect[1]
    x2 = rect[2]
    y2 = rect[3]
    if y2 >= 400:
        # print "overflow y2: ", y2
        y2 = 399
    if x2 >= 640:
        # print "overflow x2: ", x2
        x2 = 639
    if color == 'BlueViolet':
      draw_color = [226,43,138]
    if color == 'Crimson':
      draw_color = [220,20,60]
    if color == 'MediumAquamarine':
      draw_color = [0,250,154]
    if color == 'Yellow':
      draw_color = [255,255,0]
    if color == 'OrangeRed':
      draw_color = [255,69,0]
    if color == 'White':
      draw_color = [255,255,255]
    for j in range(3):
        img[y1,x1:x2,j] = draw_color[j]
        img[y2,x1:x2,j] = draw_color[j]
        img[y1:y2,x1,j] = draw_color[j]
        img[y1:y2,x2,j] = draw_color[j]

    return img

def trans2img(img):
    h, w = img.shape
    img_min = img.min()
    img_max = img.max()

    for i in range(h):
        for j in range(w):
            img[i][j] = int(((float(img[i][j]) - img_min)/(img_max-img_min))*255)
    
    temp_show = np.zeros(3*h*w)
    temp_show.resize(h,w,3)
    temp_show[:,:,0] = img.reshape([h,w]) #blue
    # cv2.imwrite(save_path, temp_show)
    return temp_show


img_path = "/home/hanyupeng/DepthData/positive/14/capture_img_7_0_0_0_depth.xml"
fs = cv2.FileStorage(img_path, cv2.FileStorage_READ)
img = fs.getNode('depth').mat()
fs.release()

draw = img.copy()
draw = trans2img(draw)

doc12 = open('/home/xingduan/YupengHan/inference/saving_docs/12/12doc.txt' ,'r')
lines = doc12.readlines()

for line in lines:
  line = line.strip('').split(' ')
  rect = [line[0], line [1], line[2], line [3]]
  draw = draw_rect(draw, rect, 'Yellow')

cv2.imwrite('/home/xingduan/YupengHan/inference/saving_docs/12/12result.jpg', draw)



