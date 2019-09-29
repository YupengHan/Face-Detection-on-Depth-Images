import sys
sys.path.append('.')
sys.path.append('/home/xingduan/caffe_parallel/python')
sys.path.append('/home/xingduan/YupengHan/caffe_modules')
import tools
import caffe
import cv2
import os
import numpy as np
import numpy.random as npr
from utils import IoU
import config

caffe.set_device(0)
caffe.set_mode_gpu()

deploy = '/home/xingduan/YupengHan/inference/models/12/12deploy.prototxt'
caffemodel = '/home/xingduan/YupengHan/inference/models/12/12deploy.caffemodel'
net_12 = caffe.Net(deploy, caffemodel, caffe.TEST)

deploy = '/home/xingduan/YupengHan/inference/models/24/24deploy.prototxt'
caffemodel = '/home/xingduan/YupengHan/inference/models/24/24deploy.caffemodel'
net_24 = caffe.Net(deploy, caffemodel, caffe.TEST)


def preprocess(img, size):
    img = cv2.resize(img,size)
    return img

def view_bar(num, total):
    rate = float(num) / total
    rate_num = int(rate * 100)
    r = '\r[%s%s]%d%%  (%d/%d)' % ("#" * rate_num, " " *
                                   (100 - rate_num), rate_num, num, total)
    sys.stdout.write(r)
    sys.stdout.flush()


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
    return temp_show

def detectFace(img_path, threshold):
    fs = cv2.FileStorage(img_path, cv2.FileStorage_READ)
    img = fs.getNode('depth').mat()
    fs.release()

    if img is None:
        print("Fail to read XML file: " + depth_path)
        return None

    origin_h, origin_w= img.shape
    scales = tools.calculateScales(img)
    out = []
    for scale in scales:
        hs = int(origin_h * scale)
        ws = int(origin_w * scale)
        net_12.blobs['data'].reshape(1, 1, hs, ws)
        net_12.blobs['data'].data[...] = preprocess(img, (ws, hs))
        out_ = net_12.forward()
        out.append(out_)
    image_num = len(scales)
    rectangles = []
    for i in range(image_num):
        cls_prob = out[i]['prob1'][0][1]
        # print "cls_prob.shape: ", cls_prob.shape
        roi = out[i]['conv4-2'][0]
        out_h, out_w = cls_prob.shape
        # print "out_h: ", out_h
        # print "out_w: ", out_w
        out_side = max(out_h, out_w)
        rectangle = tools.detect_face_12net(
            cls_prob, roi, out_side, 1 / scales[i], origin_w, origin_h, threshold[0])
        rectangles.extend(rectangle)

    # rectangles = tools.NMS(rectangles, 0.7, 'iou')

    if len(rectangles) == 0:
        print "rect drop to 0 at 12net"
        return rectangles

    net_24.blobs['data'].reshape(len(rectangles), 1, 24, 24)
    crop_number = 0
    for rectangle in rectangles:
        # rectangles_new.append(rectangular2square(rectangle, img))
        rectangle = rectangular2square(rectangle, img)
        crop_img = img[rectangle[1]:rectangle[3], rectangle[0]:rectangle[2]]
        scale_img = preprocess(crop_img, (24, 24))
        net_24.blobs['data'].data[crop_number] = scale_img
        crop_number += 1

    out = net_24.forward()
    cls_prob = out['prob1']
    roi_prob = out['ip_roi']
    rectangles = tools.filter_face_24net(cls_prob, roi_prob, rectangles, origin_w, origin_h, threshold[1])

    if len(rectangles) == 0:
        print "rect drop to 0 at 24net"
        return rectangles

    rectangles_new = []
    for rectangle in rectangles:
        rectangles_new.append(rectangular2square(rectangle, img))
    return rectangles_new


out_size = 48
labled_file = config.lable_file
im_dir = config.img_dir
base_dir = config.net48

hard_img_dir = os.path.join(base_dir, 'negative_hard')
if not os.path.exists(hard_img_dir):
  os.mkdir(hard_img_dir)

f1 = open(os.path.join(base_dir, 'hard48.txt'), 'w')

f = open(labled_file, 'r')
annos = f.readlines() # annotations, read all imag_path label(x1,y1,x2,y2)
num = len(annos)
print("%d pics in labled files" % num)

h_id = 0
image_idx = 0
threshold = [0.92, 0.5]

for anno in annos:
    anno = anno.strip().split(' ')
    anno = [x for x in anno if x != ""]
    depth_path = im_dir + anno[0]
    bbox = map(float, anno[1:])
    gts = np.array(bbox, dtype=np.float32).reshape(1,4)  # Note that box is numpy.ndarray, ([x1, y1, x2, y2])

    image_idx += 1
    view_bar(image_idx, num)

    if image_idx % 100 == 0:
        print image_idx, 'depth images is Done'

    rectangles = detectFace(depth_path, threshold)
    if rectangles is None:
        continue

    for box in rectangles:
        x_left, y_top, x_right, y_bottom, _ = box
        crop_w = x_right - x_left + 1
        crop_h = y_bottom - y_top + 1
        # ignore box that is too small or beyond image border
        if crop_w < out_size or crop_h < out_size:
            continue

        # compute intersection over union(IoU) between current box and all gt boxes
        Iou = IoU(box, gts)
        if len(Iou)==0:
            continue
        # save negative images and write label
        fs = cv2.FileStorage(depth_path, cv2.FileStorage_READ)
        img = fs.getNode('depth').mat()
        fs.release()

        if Iou[0] < 0.4:
            cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1]
            resized_im = cv2.resize(
                cropped_im, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
            # Iou with all gts must below 0.3
            save_file = os.path.join(hard_img_dir, "%d.npy" % h_id)
            f1.write("negative_hard/%d.npy 0\n" %  h_id)
            np.save(save_file, resized_im)

            h_id += 1
    print "image_idx: %d, h_id: %d" %(image_idx, h_id)
f1.close()



