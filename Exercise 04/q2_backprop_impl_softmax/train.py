import numpy as np
from Linear import Linear
from SoftMax import SoftMax
from CrossEntropy import CrossEntropy
import pickle
from read_input import read_entire_dataset
from Tanh import Tanh
from LogSoftMax import LogSoftMax
from CrossEntropyLog import CrossEntropyLog

layer1 = Linear(28 * 28, 200)
layer2 = Tanh()
layer3 = Linear(200, 10)
layer4 = LogSoftMax()
model = [layer1 ,layer2, layer3, layer4]
loss = CrossEntropyLog()
rate = 0.1
batch_size = 600
epochs = 100

data, label = read_entire_dataset('../mnist-train-data.csv', '../mnist-train-labels.csv')
data_num, _ = data.shape

for e in range(0, epochs):
    print('-------- Running epoch {} --------'.format(e))
    indices = np.random.permutation(np.arange(data_num))
    epoch_loss = 0
    for batch_num in range(0, int(np.ceil(float(data_num) / batch_size))):
        batch_indices = indices[np.arange(batch_num * batch_size, min((batch_num + 1) * batch_size, data_num))]
        batch_data = data[batch_indices, :]
        batch_label = label[batch_indices]
        z = batch_data.transpose()
        for layer in model:
            z = layer.fprop(z)
        E = loss.fprop(z, batch_label)
        epoch_loss += np.mean(E)
        dz = loss.bprop(np.tile(1 / len(batch_indices), (len(batch_indices), 1)))
        for layer in reversed(model):
            dz = layer.bprop(dz)
        for layer in model:
            layer.update(rate)
    epoch_loss /= int(np.ceil(float(data_num) / batch_size))
    print("epoch {} {}".format(e, epoch_loss))

with open('Task x Model', 'wb') as f:
    pickle.dump(model, f)
