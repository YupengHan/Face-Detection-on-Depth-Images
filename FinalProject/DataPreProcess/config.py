#! /usr/bin/env python
# coding=utf-8

import os

lable_file = "/home/xingduan/YupengHan/FinalProject/DataPreProcess/depth_label.txt"
img_dir = "/home/hanyupeng/DepthData/"
pro_img_dir = "/home/xingduan/YupengHan/FinalProject/ProcessedData"
if not os.path.exists(pro_img_dir):
  os.mkdir(pro_img_dir)
print pro_img_dir

net48 = os.path.join(pro_img_dir, "48")
if not os.path.exists(net48):
  os.mkdir(net48)

net24 = os.path.join(pro_img_dir, "24")
if not os.path.exists(net24):
  os.mkdir(net24)

net12 = os.path.join(pro_img_dir, "12")
if not os.path.exists(net12):
  os.mkdir(net12)
