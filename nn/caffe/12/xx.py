import sys

sys.path.append('.')
sys.path.append('/home/xingduan/caffe_parallel/python')
sys.path.append('/home/xingduan/YupengHan/nn/modules/12reshape')
#sys.path.append('/home/xingduan/YupengHan/nn/modules/12')
# Code need to be add to /home/xingduan/zhangQiaoYu/YP/testa/caffe to run

# code can only run on the server
# Since data layer is self written, missing the reshape fucntion,
# have to load 32 dep images each time!
import caffe 
import cv2
import numpy as np
import math

caffe.set_device(0)
caffe.set_mode_gpu()


deploy = '/home/hanyupeng/Project/testdir/12/12rdeploy.prototxt'
#caffemodel = '/home/xingduan/YupengHan/nn/caffe/12/models/12Depth_2019_0621_1315_iter_10000.caffemodel'
caffemodel = '/home/xingduan/YupengHan/nn/caffe/12/models/12Depth_2019_0625_1348_iter_10000.caffemodel'
dep_net = caffe.Net(deploy,caffemodel,caffe.TEST)


# crop_dep_img = np.load("/home/hanyupeng/Project/ProcessedData/part/0.npy")
fs = cv2.FileStorage('/home/hanyupeng/DepthData/positive/1/capture_img_1_0_0_0_depth.xml', cv2.FileStorage_READ)
depth_img = fs.getNode('depth').mat()
fs.release()

print depth_img.shape
print type(depth_img)

temp = np.zeros(400*640)
temp = temp.reshape(1,1,400,640)

dep_net.blobs['data'].reshape(1,1,400,640)
dep_net.blobs['data'].data[...] = depth_img
print "load data works"
out = dep_net.forward()
print "forward works"
cls_results = out['prob1']
print "cls_results.shape: ", cls_results.shape
# print "cls_result: ", cls_result
