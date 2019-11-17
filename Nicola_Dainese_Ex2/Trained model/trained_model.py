import numpy as np
import scipy.io # used to load dataset

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, SubsetRandomSampler
from torch.autograd import Variable

# load MNIST dataset
mat_dict = scipy.io.loadmat('MNIST.mat')
# extract the images and their labels from the dictionary
images = mat_dict['input_images']
labels = mat_dict['output_labels']

def encode_labels(labels):
    """
    Encodes a categorical variable with one hot encoding.
    """
    possible_labels = np.arange(10)
    n_samples = labels.shape[0]
    mask = np.tile(possible_labels, (n_samples,1))
    enc_labels = (mask == labels).astype('int')
    return enc_labels

enc_labels = encode_labels(labels)

class Net(nn.Module):
    
    def __init__(self, h_sizes, out_size, dropout, n_epochs, act=F.relu):
        super(Net, self).__init__()
        # Hidden layers
        self.hidden = nn.ModuleList()
        for k in range(len(h_sizes)-1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k+1]))
            self.hidden.append(nn.Dropout(dropout))
        # Output layer
        self.out_layer = nn.Linear(h_sizes[-1], out_size)
        # Activation function of the hidden layers
        self.act = act
        self.n_epochs = n_epochs
        
    def forward(self, x):
        
        # Feedforward
        for layer in self.hidden:
            x = self.act(layer(x))
        out = self.out_layer(x)
        return out
    
def eval_accuracy(net, x_test,y_test, return_predictions=False):
    # convert x and y to tensors
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)
    # create training and validation TensorDatasets
    test_set = TensorDataset(x_test,y_test)
    # create DataLoaders to load batches
    if return_predictions:
        test_loader = DataLoader(test_set, batch_size = 1)
    else:
        test_loader = DataLoader(test_set, batch_size = 16, num_workers = 2)
    correct = 0
    total = 0
    if return_predictions:
        y_pred = []
    with torch.no_grad():
        net.eval() #disable dropout
        for data in test_loader:
            images, labels = data
            outputs = net.forward(images)
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.numpy()
            if return_predictions:
                y_pred.append(predicted)
            target_labels = np.dot(labels,np.arange(10)) #decode labels
            total += len(target_labels)
            correct += (predicted == target_labels).sum()
    accuracy = 100 * correct / total
    if return_predictions:
        return accuracy, np.array(y_pred).flatten().astype('int')
    else:
        return accuracy
    
architecture = np.load('model_architecture.npy', allow_pickle=True).item()
best_net2 = Net(**architecture)
best_net2.load_state_dict(torch.load('params.pth'))
accuracy, predictions = eval_accuracy(best_net2, images,enc_labels,return_predictions=True)
print("Accuracy obtained {:.3f}%".format(accuracy))
np.savetxt('predictions.txt', predictions)