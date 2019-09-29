################################################################################################
# This part is shown cls misclassification files
################################################################################################

################################################################################################
# common part
################################################################################################

import sys
sys.path.append('.')
sys.path.append('/home/xingduan/caffe_parallel/python')
sys.path.append('/home/xingduan/YupengHan/nn/modules/')
import caffe # code can only run on the server
import cv2
import numpy as np
import math
import os

caffe.set_device(0)
caffe.set_mode_gpu()

net = 12



org_file  = open("/home/hanyupeng/Project/testdir/%d/cls_gt.txt" %net)
dep_data_dir = "/home/hanyupeng/DepthData/"

################################################################################################
# common part end
################################################################################################


if not os.path.exists('/home/xingduan/YupengHan/Rtest/%dcls_err' %(net)):
  os.mkdir('/home/xingduan/YupengHan/Rtest/%dcls_err' %(net))

cls_err_file = open("/home/xingduan/YupengHan/nn/caffe/12/cls_err_out.txt")
cls_err = cls_err_file.readlines()
err_imgs = [x.strip('').split(" ") for x in cls_err]
err_imgs = [x[0] for x in err_imgs]
print "err_imgs[0]: ", err_imgs[0]


data = org_file.readlines()
data = [x.strip('').split(" ") for x in data]
cls_imgs = [x[0] for x in data]
# print "roi_imgs[0]: ", roi_imgs[0]
org_xmls = [dep_data_dir + x[2] for x in data]
# print "org_xmls[0]: ", org_xmls[0]
img_locs = [[x[3],x[4],x[5],x[6]] for x in data]
# print "img_locs[0]: ", img_locs[0]
face_gt  = [[x[8],x[9],x[10],x[11].strip('\n')] for x in data]

totalnum = len(data)


for i in range(len(err_imgs)):
  depth_crop = np.load("/home/hanyupeng/Project/testdir/%d/datadir/" %net + err_imgs[i])
  print "depth_crop.shape: ",depth_crop.shape
  print "err_imgs[i]: ",err_imgs[i]
  depth_min = depth_crop.min()
  print "depth_min: ",depth_min
  depth_max = depth_crop.max()
  print "depth_max: ",depth_max

  for m in range(12):
    for n in range(12):
      depth_crop[m][n] = int(((float(depth_crop[m][n]) - depth_min)/(depth_max-depth_min))*255)
  
  crop_show = np.zeros(3*12*12)
  crop_show.resize(12,12,3)
  crop_show[:,:,0] = depth_crop.reshape([12,12]) #blue
  cv2.imwrite('/home/xingduan/YupengHan/Rtest/%dcls_err/%derr_input.png' %(net, i), crop_show)


  for j in range(totalnum):
    if cls_imgs[j] == err_imgs[i]:
      print "j: ", j

      fs = cv2.FileStorage(org_xmls[j], cv2.FileStorage_READ)
      org_img = fs.getNode('depth').mat()
      fs.release()
      # print "org_img.shape: ", org_img.shape

      
      depth_min = org_img.min()
      depth_max = org_img.max()

      for m in range(400):
        for n in range(640):
          org_img[m][n] = int(((float(org_img[m][n]) - depth_min)/(depth_max-depth_min))*255)

      temp_show = np.zeros(3*400*640)
      temp_show.resize(400,640,3)
      temp_show[:,:,0] = org_img.reshape([400,640]) #blue

      size = int(img_locs[j][2]) - int(img_locs[j][0])
      # Green crop_img_loc
      x = int(img_locs[j][0])
      y = int(img_locs[j][1])
      for m in range(size):
        temp_show[y,x+m,1] = 255 #green
        temp_show[y+size-1,x+m,1] = 255 #green
      for m in range(size):
        temp_show[y+m,x,1] = 255 #green
        temp_show[y+m,x+size-1,1] = 255 #green
      
      # Red ground_truth_loc
      face_gt[j] = map(int, face_gt[j])
      x1 = face_gt[j][0]
      y1 = face_gt[j][1]
      x2 = face_gt[j][2]
      y2 = face_gt[j][3]
      temp_show[y1,x1:x2,2] = 255 #red
      temp_show[y2,x1:x2,2] = 255 #red
      temp_show[y1:y2,x1,2] = 255 #red
      temp_show[y1:y2,x2,2] = 255 #red

      cv2.imwrite('/home/xingduan/YupengHan/Rtest/%dcls_err/%derr.png' %(net, i), temp_show)
          







