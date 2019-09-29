import sys
sys.path.append('.')
sys.path.append('/home/xingduan/caffe_parallel/python')
sys.path.append('/home/xingduan/YupengHan/nn/modules/24')
import caffe # code can only run on the server
import cv2
import numpy as np
import math

caffe.set_device(0)
caffe.set_mode_gpu()

size = 24

deploy = '/home/hanyupeng/Project/testdir/24/24deploy.prototxt'
caffemodel = '/home/xingduan/YupengHan/nn/caffe/24/models/24Depth_2019_0704_1114_iter_15000.caffemodel'
dep_net = caffe.Net(deploy,caffemodel,caffe.TEST)

# f = "/home/hanyupeng/Project/testdir/roi_gt.txt"
# f = "/home/hanyupeng/Project/testdir/cls_gt.txt"
cls_file  = open("/home/hanyupeng/Project/testdir/24/cls_test.txt")
# roi_file  = open("/home/hanyupeng/Project/testdir/roi_test.txt")
# cls_label = open("/home/hanyupeng/Project/testdir/cls_gt.txt")
cls_out       = open("/home/xingduan/YupengHan/nn/caffe/24/clstestout.txt", 'w')
cls_err_out   = open("/home/xingduan/YupengHan/nn/caffe/24/cls_err_out.txt", 'w')


cls_data = cls_file.readlines()
cls_data = [x.strip("").split(" ") for x in cls_data]
img_data = [x[0] for x in cls_data]
# print "img_data[0]: ", img_data[0]
label_data = [x[1] for x in cls_data]
# print "label_data[0]: ", label_data[0]

# roi_data = roi_file.readlines()
# roi_data = [x.strip("").split(" ") for x in roi_data]
# roi_imgs = [x[0] for x in roi_data]


num = len(cls_data)
correct_count = 0

print "%d pics in cls data" % num

for i in range(num):
  depth_img = np.load("/home/hanyupeng/Project/testdir/24/datadir/" + img_data[i])
  depth_img = depth_img.reshape(1,1,size,size)
  dep_net.blobs['data'].reshape(1,1,size,size)
  dep_net.blobs['data'].data[...] = depth_img
  out = dep_net.forward()
  cls_result = out['prob1']
  if cls_result[0][1] > 0.5:
    cls_pred = 1
  else:
    cls_pred = 0

  if cls_pred == int(label_data[i]):
    correct_count += 1
  else:
    cls_err_out.write(img_data[i] + " %f, %f, %d\n" %(cls_result[0][0], cls_result[0][1], cls_pred))

  cls_out.write(img_data[i] + " %f, %f, %d\n" %(cls_result[0][0], cls_result[0][1], cls_pred))  
  

rate = float(correct_count)/(num)
print "num: %d, correct_count: %d, rate: %f" %(num, correct_count, rate)


