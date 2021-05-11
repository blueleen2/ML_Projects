
from math import sqrt
import pandas as pd


data = pd.read_csv("./TestDataForKNN.csv")
lists = []

def csv2list(filename):
	import csv
	file = open(filename,'r')
	csvfile = csv.reader(file)
	lists = []
	Ignore_row = 1;
	i = 0
	for item in csvfile:
		if i > Ignore_row:
			convert_item = list(map(float, item))
			lists.append(convert_item)
		i+=1
	return lists

'''
dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]
'''

# data .csv파일로 저장하기
'''
df = pd.DataFrame(dataset)
df.to_csv('TestDataForKNN.csv',index=False, encoding='cp949')
#encoding 옵션: csv 파일에서 한글 (컬럼 혹은 내용) 읽어올 때 encoding='cp949' (혹은 encoding='euc-kr') 옵션 사용, 엑셀의 csv 파일 utf-8 인코딩 (유니코드) 인식 버그
#0 : x좌표, 1: y: 좌표, 2: 레이블
'''

def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance +=(row1[i]-row2[i])**2
	return sqrt(distance)

def get_neighbors(train, test_row, num_neighbors):
	distances = list()
	for train_row in train:
		dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	# Key 값을 기준으로 오르차순 정렬(dist 기준으로 정렬)
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors

def predict_classification(train, test_row, num_neighbors):
	neighbors = get_neighbors(train, test_row, num_neighbors)
	for neighbor in neighbors:
		print(neighbor)
	output_values = [row[-1] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)
	return prediction



'''
for row in dataset:
	distance = euclidean_distance(row0, row)
	print(distance)
'''
'''
neighbors = get_neighbors(dataset, row0, 3)
for neighbor in neighbors:
	print(neighbor)
'''


if __name__ == '__main__':
	TrainData = csv2list("./TestDataForKNN.csv")
	# 새로운 입력값
	row0 = [6, 6, 0.0]
	prediction = predict_classification(TrainData, row0, 3)
	print("Expected {}, Got {}".format(row0[-1], prediction))

