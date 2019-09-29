import caffe # code can only run on the server
import cv2
import numpy as np

caffe.set_device(0)
caffe.set_mode_gpu()

deploy = '/home/xingduan/zhangQiaoYu/YP/testa/caffe/test_train_val.prototxt'
caffemodel = '/home/xingduan/zhangQiaoYu/YP/testa/caffe/models/Depth_2019_0613_iter_10000.caffemodel'
dep_net = caffe.Net(deploy,caffemodel,caffe.TEST)

crop_dep_img = "/home/hanyupeng/Project/ProcessedData/part/0.npy"
# part/0.npy 0.10 -0.31 0.26 0.28
print crop_dep_img.shape
dep_net.blobs['data'].reshape(1, 1, 48, 48)
dep_net.blobs['data'].data[...] = crop_dep_img
out = dep_net.forward()
print "out: ", out
