# with open('list.txt', 'r') as f, open('list1.txt','w') as f1:
# 	for line in f:
# 		item = line.split('||')
# 		line = item[0][1:]+'||'+item[1][1:]
# 		f1.write(line)

# import math
# size = 256
# for i in range(8):
# 	size = math.ceil((size+2-3)*1.0/2)
# 	print size






# with open('list_test.txt', 'r') as f, open('list_test1.txt','w') as f1:
# 	for line in f:
# 		line = line.strip()
# 		line = line.split('||')
# 		f1.write(line[0]+'\r\n')

# print '======Finished!======='


import random
with open('list.txt', 'r') as f, open('list_train.txt', 'w') as f1, open('list_test.txt', 'w') as f2:
	alldata = []
	for line in f:
		alldata.append(line)
	random.shuffle(alldata)
	lens = len(alldata)
	threshold = int(lens*0.8)
	for i in range(lens):
		if i < threshold:
			f1.write(alldata[i])
		else:
			f2.write(alldata[i].split('||')[0]+'\r\n')
print 'Finished'



