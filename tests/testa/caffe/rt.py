import sys
sys.path.append('.')
sys.path.append('/home/xingduan/caffe_parallel/python')
sys.path.append('/home/xingduan/zhangQiaoYu/YP/testa/modules/')
import caffe # code can only run on the server
import cv2
import numpy as np

caffe.set_device(0)
caffe.set_mode_gpu()
print 11

deploy = '/home/xingduan/zhangQiaoYu/YP/testa/caffe/test_train_val.prototxt'
caffemodel = '/home/xingduan/zhangQiaoYu/YP/testa/caffe/models/Depth_2019_0613_iter_10000.caffemodel'
dep_net = caffe.Net(deploy,caffemodel,caffe.TEST)
print 16

crop_dep_img = np.load("/home/hanyupeng/Project/ProcessedData/48/part/0.npy")
# part/0.npy 0.10 -0.31 0.26 0.28
print crop_dep_img.shape
print type(crop_dep_img)
crop_dep_img.reshape(1,48,48)
temp = np.zeros(32*48*48)
temp = temp.reshape(32,1,48,48)
for i in range(32):
  temp[i] = crop_dep_img

dep_net.blobs['roi_img_data'].reshape(32, 1, 48, 48)
dep_net.blobs['roi_img_data'].data[...] = temp
out = dep_net.forward()
print "out: ", out
relative_pos = dep_net.blobs['ip_roi'].data[0].flatten()
print "relative_pos: ", relative_pos

cls_result = dep_net.blobs['ip_cls'].data[0].flatten()
print "cls_result: ", cls_result
