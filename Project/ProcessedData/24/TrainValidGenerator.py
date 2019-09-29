import os
import math
import numpy as np
import numpy.random as npr

def readlist(fp):
    list_tmp = []
    fin = open(fp, 'r')
    while True:
        line = fin.readline()
        if line:
            list_tmp.append(line)
        else:
            break
    return list_tmp

pos_file = "/home/hanyupeng/Project/ProcessedData/24/pos.txt"
# neg_file = "/home/hanyupeng/Project/ProcessedData/24/neg.txt"
hard_file = "/home/hanyupeng/Project/ProcessedData/24/hard.txt"

if not os.path.exists("/home/hanyupeng/Project/ProcessedData/24/trainValid"):
    os.mkdir("/home/hanyupeng/Project/ProcessedData/24/trainValid")

f_cls_train = open("/home/hanyupeng/Project/ProcessedData/24/trainValid/cls_train.txt", 'w')
f_cls_valid = open("/home/hanyupeng/Project/ProcessedData/24/trainValid/cls_val.txt", 'w')

pos_data = readlist(pos_file)
npr.shuffle(pos_data)
num = len(pos_data)
trainNum = int(math.floor(0.9 * num))
temp_train = pos_data[num- trainNum: num]
temp_valid = pos_data[0: num- trainNum]
neg_data = readlist(hard_file)
npr.shuffle(neg_data)
numn = len(neg_data)
if numn > 5 * num:
    numn = 5 * num
else :
    print("not enough hard neg!")
neg_data = neg_data[0:numn]
trainNum = int(math.floor(0.9 * numn))
temp_train = temp_train + neg_data[numn - trainNum: numn]
temp_valid = temp_valid + neg_data[0: numn - trainNum]
print(len(temp_train))
print(len(temp_valid))
npr.shuffle(temp_train)
npr.shuffle(temp_valid)
f_cls_train.write("".join(temp_train).strip('\n'))
f_cls_valid.write("".join(temp_valid).strip('\n'))
f_cls_train.close()
f_cls_valid.close()

part1 = "/home/hanyupeng/Project/ProcessedData/24/part1.txt"
part2 = "/home/hanyupeng/Project/ProcessedData/24/part2.txt"

f_roi_train = open("/home/hanyupeng/Project/ProcessedData/24/trainValid/roi_train.txt", 'w')
f_roi_valid = open("/home/hanyupeng/Project/ProcessedData/24/trainValid/roi_val.txt", 'w')

part = [x for x in readlist(part1) if x != "\n"]
npr.shuffle(part)
pos  = [x for x in readlist(part2) if x != "\n"]
npr.shuffle(pos)


num = int(len(pos)/2)
trainNum = int(math.floor(0.9 * num))
temp_train = pos[0: trainNum]

temp_valid = pos[trainNum: num]

temp_train = temp_train + part[0: trainNum]
print("temp_train: ", len(temp_train))
temp_valid = temp_valid + part[trainNum: num]
print("temp_valid: ", len(temp_valid))
npr.shuffle(temp_train)
npr.shuffle(temp_valid)
f_roi_train.write("".join(temp_train).strip('\n'))
f_roi_valid.write("".join(temp_valid).strip('\n'))
f_roi_train.close()
f_roi_valid.close()

