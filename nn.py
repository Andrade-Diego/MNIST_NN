import numpy as np
import csv

ALPHA = .01			#learning rate
LAYERS = 1				#arbitrary number layers
NODES = 200				#arbitrary number nodes in layer
POSSIBLE_OUTPUTS = 5	#num of possible outputs
NUM_INPUTS = 784
LABEL_DICT = {	0: [1, 0, 0, 0, 0],
				1: [0, 1, 0, 0, 0],
				2: [0, 0, 1, 0, 0],
				3: [0, 0, 0, 1, 0],
				4: [0, 0, 0, 0, 1]}

def fileReader(path):
	labels = []
	data = []
	with open(path, newline = '') as dataFile:
		reader = csv.reader(dataFile, delimiter = ',')
		shuffleData = list()
		for row in reader:
			shuffleData.append(row)
		np.random.shuffle(shuffleData)
		for row in shuffleData:
			labels.append(int(row[0]))
			data.append([int(row[i]) / 255 for i in range(1,len(row))])
	return (np.array(labels), np.array(data))

class Layer:
	def __init__(self, numInputs, numberNodes = 20):
		self.numNodes = numberNodes
		self.weights = (2 * np.random.random((numInputs, self.numNodes)) - 1)/100

	def calculateWeightedSum(self, inputs):
		self.weightedSum = np.dot(inputs, self.weights)

	def calculateActivation(self):
		self.activation = sigmoid(self.weightedSum)

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def dSigmoid(x):
	return sigmoid(x) * (1.0 - sigmoid(x))

def forwardPropagate(x, hLayer, oLayer):
	hLayer.calculateWeightedSum(x)
	hLayer.calculateActivation()
	oLayer.calculateWeightedSum(hLayer.activation)
	return oLayer.weightedSum

def backPropagate(x, y, hLayer, oLayer):
	oError = (2*(oLayer.weightedSum - LABEL_DICT[y]))
	hError = hLayer.activation * (1 - hLayer.activation) * np.dot(oError, oLayer.weightedSum.T)
	hPartial = x[np.newaxis, :, :] * hError[: , np.newaxis, :]

	oPartial = hLayer.activation * oError.T
	return hPartial, oPartial

def runNet(data):
	Y = data[0]
	X = data[1]
	hLayer = Layer(len(X[0]) + 1, NODES)
	oLayer = Layer(hLayer.numNodes, POSSIBLE_OUTPUTS)
	for iteration in range (0, 10):
		print("iteration number: ", iteration)
		for i in range (0, len(Y)):
			hPartialAvg = 0
			oPartialAvg = 0
			for j in range (0, 100):
				if i + j >= len(Y):
					break
				x = np.reshape(np.hstack(([1], X[i])), (785,1))
				forwardPropagate(x.T, hLayer, oLayer)
				hPartial, oPartial = backPropagate(x, Y[i], hLayer, oLayer)

				hPartialAvg += np.average(hPartial, axis = 0)
				oPartialAvg += oPartial
			hPartialAvg /= j
			oPartialAvg /= j

			hLayer.weights -= ALPHA * hPartialAvg
			oLayer.weights -= ALPHA * oPartialAvg.T
			i = i + j
			if i >= len(Y):
				break
	return hLayer, oLayer

def testNet (hLayer, oLayer, testData):
	X = testData[1]
	Y = testData[0]
	accuracy = 0
	for i in range(0, len(Y)):
		x = np.reshape(np.hstack(([1], X[i])), (785,1))
		prediction = (forwardPropagate(x.T, hLayer, oLayer))
		print("prediction is ", prediction)
		yHat = np.argmax(prediction)
		if yHat == Y[i]:
			accuracy += 1
	accuracy /= len(Y)
	return accuracy

if __name__ == "__main__":
	```
	trainData1 = fileReader("./mnist_train_0_1.csv")
	testData1 = fileReader("./mnist_test_0_1.csv")
	trainData2 = fileReader("./mnist_train_0_4.csv")
	testData2 = fileReader("./mnist_test_0_4.csv")
	net1 = runNet(trainData1)
	print(testNet(net1[0], net1[1], testData1))
	net2 = runNet(trainData2)
	#print("outlayer info: ", net[1].activation, net[1].weights)
	print(testNet(net2[0], net2[1], testData2))
	```
	print("hello world I'm learning to use git on the terminal")
