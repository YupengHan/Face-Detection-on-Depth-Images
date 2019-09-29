import sys
sys.path.append('.')
sys.path.append('/home/xingduan/caffe_parallel/python')
sys.path.append('/home/xingduan/YupengHan/caffe_modules')
import tools
import caffe
import cv2
import numpy as np

caffe.set_device(0)
caffe.set_mode_gpu()

deploy = '/home/xingduan/YupengHan/inference/models/12/12deploy.prototxt'
caffemodel = '/home/xingduan/YupengHan/inference/models/12/12deploy.caffemodel'
net_12 = caffe.Net(deploy, caffemodel, caffe.TEST)

deploy = '/home/xingduan/YupengHan/inference/models/24/24deploy.prototxt'
caffemodel = '/home/xingduan/YupengHan/inference/models/24/24deploy.caffemodel'
net_24 = caffe.Net(deploy, caffemodel, caffe.TEST)

deploy = '/home/xingduan/YupengHan/inference/models/48/48deploy.prototxt'
caffemodel = '/home/xingduan/YupengHan/inference/models/48/48deploy.caffemodel'
net_48 = caffe.Net(deploy, caffemodel, caffe.TEST)



def preprocess(img, size):
    # print "size:", size
    img = cv2.resize(img,size)
    # print "img.shape: ",img.shape
    return img


def rectangular2square(rectangle, caffe_img):
    x1 = rectangle[0]
    y1 = rectangle[1]
    x2 = rectangle[2]
    y2 = rectangle[3]
    h, w = caffe_img.shape
    size = ((x2 - x1) + (y2 - y1)) * 0.5
    x_c = (x1 + x2) * 0.5
    y_c = (y1 + y2) * 0.5
    rectangle[0] = int(round(max(0, x_c - size * 0.5)))
    rectangle[1] = int(round(max(0, y_c - size * 0.5)))
    rectangle[2] = int(round(min(w, x_c + size * 0.5)))
    rectangle[3] = int(round(min(h, y_c + size * 0.5)))
    return rectangle

# def save_pred_img(img, rect, save_path):
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

def detectFace(img_path, threshold):
    # img = cv2.imread(img_path)
    fs = cv2.FileStorage(img_path, cv2.FileStorage_READ)
    img = fs.getNode('depth').mat()
    fs.release()
    origin_h, origin_w= img.shape
    scales = tools.calculateScales(img)
    out = []
    for scale in scales:
        hs = int(origin_h * scale)
        ws = int(origin_w * scale)
        net_12.blobs['data'].reshape(1, 1, hs, ws)
        # print "hs: ",hs
        # print "ws: ",ws
        net_12.blobs['data'].data[...] = preprocess(img, (ws, hs))
        out_ = net_12.forward()
        out.append(out_)
    image_num = len(scales)
    rectangles = []
    for i in range(image_num):
        cls_prob = out[i]['prob1'][0][1]
        roi = out[i]['conv4-2'][0]
        out_h, out_w = cls_prob.shape
        out_side = max(out_h, out_w)
        rectangle = tools.detect_face_12net(
            cls_prob, roi, out_side, 1 / scales[i], origin_w, origin_h, threshold[0])
        rectangles.extend(rectangle)

    rectangles = tools.NMS(rectangles, 0.7, 'iou')

    if len(rectangles) == 0:
        print "rect drop to 0 at 12net"
        return rectangles

    doc12 = open('/home/xingduan/YupengHan/inference/saving_docs/12/12doc.txt', 'w')
    for temp_rectangle in rectangles:
        doc12.write('%d %d %d %d %f\n' %(temp_rectangle[0], temp_rectangle[1], temp_rectangle[2], temp_rectangle[3], temp_rectangle[4]))
    

    # Here might be a problme
    net_24.blobs['data'].reshape(len(rectangles), 1, 24, 24)
    crop_number = 0
    for rectangle in rectangles:
        rectangle = rectangular2square(rectangle, img)
        crop_img = img[int(rectangle[1]):int(
            rectangle[3]), int(rectangle[0]):int(rectangle[2])]
        net_24.blobs['data'].data[crop_number] = preprocess(crop_img, (24, 24))
        crop_number += 1
    # Here might be a problme
    out = net_24.forward()
    cls_prob = out['prob1']
    roi_prob = out['ip_roi']
    rectangles = tools.filter_face_24net(
        cls_prob, roi_prob, rectangles, origin_w, origin_h, threshold[1])

    if len(rectangles) == 0:
        print "rect drop to 0 at 24net"
        return rectangles
    net_48.blobs['data'].reshape(len(rectangles), 1, 48, 48)
    crop_number = 0
    for rectangle in rectangles:
        rectangle = rectangular2square(rectangle, img)
        crop_img = img[int(rectangle[1]):int(
            rectangle[3]), int(rectangle[0]):int(rectangle[2])]
        net_48.blobs['data'].data[crop_number] = preprocess(crop_img, (48, 48))
        crop_number += 1
    out = net_48.forward()
    cls_prob = out['prob1']
    roi_prob = out['ip_roi']
    rectangles = tools.filter_face_48net(
        cls_prob, roi_prob, rectangles, origin_w, origin_h, threshold[2])

    return rectangles


threshold = [0.93, 0.8, 0.9]
img_path = "/home/hanyupeng/DepthData/positive/165/capture_img_8_0_0_4_depth.xml"
# /home/hanyupeng/DepthData/positive/165/capture_img_8_0_0_4_depth.xml
rectangles = detectFace(img_path, threshold)
fs = cv2.FileStorage(img_path, cv2.FileStorage_READ)
img = fs.getNode('depth').mat()
fs.release()

draw = img.copy()
draw = trans2img(draw)

for i in range(len(rectangles)):
    # save_path = '/%d.png' %i 
    draw = draw_rect(draw, rectangles[i],'BlueViolet')

cv2.imwrite('/home/xingduan/YupengHan/inference/result.jpg', draw)



