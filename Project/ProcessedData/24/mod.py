
f = open('/home/hanyupeng/Project/ProcessedData/24/hard.txt' , 'r')
data = f.readlines()
print "data[0][3:4]: ", data[0][3:4]
data = [x[3:] for x in data]
out = open('/home/hanyupeng/Project/ProcessedData/24/hardnew.txt', 'w')
out.write("".join(data).strip('\n'))
out.close()





