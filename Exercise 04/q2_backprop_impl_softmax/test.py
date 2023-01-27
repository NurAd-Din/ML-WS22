import numpy as np
import pickle
from read_input import read_entire_dataset
# from PIL import Image
#
# img = Image.open('.jpg').convert('L')
# img.thumbnail((28,28), Image.ANTIALIAS)
# arr = np.asarray(img).flatten().reshape((28*28, 1)) / 255
#
# z = arr
# for layer in model:
#     z = layer.fprop(z)

with open('Task x Model', 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    model = u.load()


data, label = read_entire_dataset('../cifar10-valid-data.csv', '../cifar10-valid-labels.csv')
data_num, _ = data.shape

batch_size = 100
accurate_pred = 0
for batch_num in range(0, int(np.ceil(data.shape[0] / float(batch_size)) + 0.1)):
    batch_indices = np.arange(batch_num * batch_size, min((batch_num + 1) * batch_size, data.shape[0]))
    batch_data = data[batch_indices, :]
    batch_label = label[batch_indices]
    z = batch_data.transpose()
    for module in model:
        z = module.fprop(z)
    predictions = np.argmax(z, axis=0)
    accurate_batch_pred = np.isclose(predictions, batch_label)
    accurate_pred += np.sum(accurate_batch_pred)
print("Errors:", data.shape[0] - accurate_pred)
print("Test accuracy {}".format((float(accurate_pred) / data.shape[0])))