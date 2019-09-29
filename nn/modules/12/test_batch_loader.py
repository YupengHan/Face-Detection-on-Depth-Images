from cls_batch_loader import *
import datetime

params = {}
params['root_folder'] = '/home/hanyupeng/Project/ProcessedData/'
params['source'] = '/home/hanyupeng/testa/cls_t.txt'
params['batch_size'] = 256
params['shuffle'] = True
params['rotate'] = True
params['rotate_range'] = 20
params['color_jitter'] = False
params['color_jitter_prob'] = 5
params['mirror'] = True

batch_loader = BatchLoader(params)
# batch_loader.show_img = True
#while True:
for i in range(1,5):
	starttime = datetime.datetime.now()
	# print(starttime)
	batch_loader.load_next_image()
	endtime = datetime.datetime.now()
	# print(endtime)
	print((endtime - starttime))

