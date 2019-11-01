#@author : Nicola Dainese
#@date : 01-10-2019

import numpy as np

from scipy.special import expit
# expit(x) = 1/(1+exp(-x))
def sigmoid(x):
    return expit(x)

# 1st derivative
#sigmoid_der = lambda x: sigmoid(x) * (1 - sigmoid(x))

def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))

# vectorized implementation

def ReLU(x):
    x = x.reshape((len(x),1))
    zeros = np.zeros(x.shape)
    z = np.stack((x,zeros), axis=1)
    return np.max(z, axis=1)

def ReLU_der(x):
    z = np.zeros(x.shape)
    z[x>0] = 1
    return z

# vectorized implementation

def LeakyReLU(x):
    x = x.reshape((len(x),1))
    ones = x*0.1
    z = np.stack((x,ones), axis=1)
    return np.max(z, axis=1)

def LeakyReLU_der(x):
    z = np.ones(x.shape)
    z[x<0] = 0.1
    return z

#%% Network class

class Network():
    
    def __init__(self, N_neurons, act_func, act_der_func, lr, n_epochs, \
                 en_decay=True, lr_final=1e-4, early_stopping = True, tol = 1e-3, \
                 en_penalty=False, penalty=1e-4, en_grad_clipping=True, grad_treshold=10):
        
        Ni, Nh1, Nh2, No = N_neurons
        ### WEIGHT INITIALIZATION (Xavier)
        # Initialize hidden weights and biases (layer 1)
        Wh1 = (np.random.rand(Nh1, Ni) - 0.5) * np.sqrt(12 / (Nh1 + Ni))
        Bh1 = np.zeros([Nh1, 1])
        self.WBh1 = np.concatenate([Wh1, Bh1], 1) # Weight matrix including biases
        # Initialize hidden weights and biases (layer 2)
        Wh2 = (np.random.rand(Nh2, Nh1) - 0.5) * np.sqrt(12 / (Nh2 + Nh1))
        Bh2 = np.zeros([Nh2, 1])
        self.WBh2 = np.concatenate([Wh2, Bh2], 1) # Weight matrix including biases
        # Initialize output weights and biases
        Wo = (np.random.rand(No, Nh2) - 0.5) * np.sqrt(12 / (No + Nh2))
        Bo = np.zeros([No, 1])
        self.WBo = np.concatenate([Wo, Bo], 1) # Weight matrix including biases
        
        ### ACTIVATION FUNCTION
        self.act = act_func
        self.act_der = act_der_func
        
        # TRAINING PARAMETERS
        self.lr = lr
        self.n_epochs = n_epochs
        self.en_decay = en_decay
        self.tolerance = tol
        if en_decay == True:
            self.lr_decay = (lr_final / lr)**(1 / n_epochs)
        else:
            self.lr_decay = None
        
        if early_stopping == True:
            self.early_stopping = True
        else:
            self.early_stopping = False
        
        if (en_penalty == 'L1'):
            self.penalty = penalty
            self.penalty_type = 'L1'
        elif (en_penalty == 'L2'):
            self.penalty = penalty
            self.penalty_type = 'L2'
        else:
            self.penalty = None
            self.penalty_type = None
            
        if en_grad_clipping == True:
            self.grad_treshold = grad_treshold
        else:
            self.grad_treshold = None
    
    def load_weights(self, WBh1, WBh2, WBo):
        self.WBh1 = WBh1
        self.WBh2 = WBh2
        self.WBo = WBo
        
    def save_weights(self):
        np.save("WBh1", self.WBh1)
        np.save("WBh2", self.WBh2)
        np.save("WBo", self.WBo)
        
    def forward(self, x, additional_out=False):
        
        # Convert to numpy array
        x = np.array(x)
        
        ### Hidden layer 1
        # Add bias term
        X = np.append(x, 1)
        # Forward pass (linear)
        H1 = np.matmul(self.WBh1, X)
        # Activation function
        Z1 = self.act(H1)
        
        ### Hidden layer 2
        # Add bias term
        Z1 = np.append(Z1, 1)
        # Forward pass (linear)
        H2 = np.matmul(self.WBh2, Z1)
        # Activation function
        Z2 = self.act(H2)
        
        ### Output layer
        # Add bias term
        Z2 = np.append(Z2, 1)
        # Forward pass (linear)
        Y = np.matmul(self.WBo, Z2)
        # NO activation function
        
        if additional_out:
            return Y.squeeze(), Z2
        
        return Y.squeeze()
        
    def update(self, x, label):
        
        # Convert to numpy array
        X = np.array(x)
        
        ### Hidden layer 1
        # Add bias term
        X = np.append(X, 1)
        # Forward pass (linear)
        H1 = np.matmul(self.WBh1, X)
        # Activation function
        Z1 = self.act(H1)
        
        ### Hidden layer 2
        # Add bias term
        Z1 = np.append(Z1, 1)
        # Forward pass (linear)
        H2 = np.matmul(self.WBh2, Z1)
        # Activation function
        Z2 = self.act(H2)
        
        ### Output layer
        # Add bias term
        Z2 = np.append(Z2, 1)
        # Forward pass (linear)
        Y = np.matmul(self.WBo, Z2)
        # NO activation function
        
        # Evaluate the derivative terms
        D1 = Y - label
        D2 = Z2
        D3 = self.WBo[:,:-1]
        D4 = self.act_der(H2)
        D5 = Z1
        D6 = self.WBh2[:,:-1]
        D7 = self.act_der(H1)
        D8 = X
        
        # Layer Error
        Eo = D1
        Eh2 = np.matmul(Eo, D3) * D4
        Eh1 = np.matmul(Eh2, D6) * D7
         
        # Derivative for weight matrices
        dWBo = np.matmul(Eo.reshape(-1,1), D2.reshape(1,-1))
        dWBh2 = np.matmul(Eh2.reshape(-1,1), D5.reshape(1,-1))
        dWBh1 = np.matmul(Eh1.reshape(-1,1), D8.reshape(1,-1))
        
        if self.grad_treshold != None:
            # compute the norm of the whole gradient
            grad_norm = np.linalg.norm(np.concatenate([dWBo.flatten(),dWBh2.flatten(),dWBh1.flatten()]))
            # if the norm exceeds the threshold, we rescale component wise so that the norm is clipped to the treshold
            if grad_norm > self.grad_treshold :
                dWBo = dWBo/grad_norm*self.grad_treshold 
                dWBh1 = dWBh1/grad_norm*self.grad_treshold 
                dWBh2 = dWBh2/grad_norm*self.grad_treshold 
      
        if self.penalty_type == 'L1':
            
            # Compute penalty 
            WBh1_sgn = np.ones(self.WBh1.shape)
            WBh1_sgn[self.WBh1<0] = -1

            WBh2_sgn = np.ones(self.WBh2.shape)
            WBh2_sgn[self.WBh2<0] = -1

            WBo_sgn = np.ones(self.WBo.shape)
            WBo_sgn[self.WBo<0] = -1
        
            # Update the weights
            self.WBh1 -= self.lr * (dWBh1 + self.penalty * WBh1_sgn)
            self.WBh2 -= self.lr * (dWBh2 + self.penalty * WBh2_sgn)
            self.WBo -= self.lr * (dWBo + self.penalty * WBo_sgn)
            
            # Compute the loss
            loss = (Y - label)**2/2 + \
                   self.penalty * (np.abs(self.WBh1).sum() + np.abs(self.WBh2).sum() + np.abs(self.WBo).sum())
            
        elif self.penalty_type == 'L2':
        
            # Update the weights
            self.WBh1 -= self.lr * (dWBh1 + self.penalty * self.WBh1)
            self.WBh2 -= self.lr * (dWBh2 + self.penalty * self.WBh2)
            self.WBo -= self.lr * (dWBo + self.penalty * self.WBo)
            
            # Compute the loss
            loss = (Y - label)**2/2 + \
                   self.penalty *( (self.WBh1**2).sum() + (self.WBh2**2).sum() + (self.WBo**2).sum() )
        
        else: # no penalty case
            
            # Update the weights
            self.WBh1 -= self.lr * dWBh1
            self.WBh2 -= self.lr * dWBh2
            self.WBo -= self.lr * dWBo
            
            # Compute penalty error
            loss = (Y - label)**2/2     
            
        return loss
    
    def train(self, x_train, y_train, x_val, y_val, train_log=False, verbose=False):
    
        if (train_log == True) or (self.early_stopping == True):
            train_loss_log = []
            val_loss_log = []
        if (self.early_stopping == True):
            last_check = 40
        for num_ep in range(self.n_epochs):
            # Learning rate decay
            if self.en_decay:
                self.lr *= self.lr_decay
            # Train single epoch (sample by sample, no batch for now)
            train_loss_vec = [self.update(x, y) for x, y in zip(x_train, y_train)]
            avg_train_loss = np.mean(train_loss_vec)
            # Validation network
            y_val_est = np.array([self.forward(x) for x in x_val])
            avg_val_loss = np.mean((y_val_est - y_val)**2/2) # just mean squared error
            # Log
            train_loss_log.append(avg_train_loss)
            val_loss_log.append(avg_val_loss)
            
            if verbose == True:
                print('Epoch %d - lr: %.5f - Train loss: %.4f - Test loss: %.4f' % \
                      (num_ep + 1, self.lr, avg_train_loss, avg_val_loss))
            
            if self.early_stopping and (num_ep > 100):
                # if the minimum loss of the last 20 epochs is greater than the mean of the previous 20 epochs
                # in a range of tolerance specified, then either stop if the learning is adaptive
                # or adapte the learning rate untill it is smaller or equal to 10^-4
                if np.mean(val_loss_log[-40:-20]) - self.tolerance < np.min(val_loss_log[-20:]):
                    if (not self.en_decay) and (self.lr >= 1e-4): #only case in which we do not stop
                        if num_ep - last_check > 20: #wait at least 20 epochs between adjustment of learning rates
                            if verbose:
                                print("Training is not improving. Reducing by 10 the learning rate.")
                            self.lr = self.lr/10 
                            last_check = num_ep
                        else:
                            continue
                    else:
                        if (not self.en_decay) and num_ep - last_check < 20:
                            continue
                        else:
                            if verbose:
                                print("Training interrupted by early stopping.")
                            break 
                else:
                    continue
            
        if train_log == True:
            return train_loss_log, val_loss_log
    
    
    def evaluate_mean_loss(self, x_test, y_test, save=False):
        y_predicted = np.array([self.forward(x) for x in x_test])
        average_loss = np.mean((y_predicted - y_test)**2/2)
        if save:
            np.savetxt('predictions.txt', y_predicted)
        return average_loss
        
    def plot_weights(self):
        fig, axs = plt.subplots(3, 1, figsize=(12,9))
        axs[0].hist(self.WBh1.flatten(), 20)
        axs[0].set_title('First hidden layer weights')
        axs[1].hist(self.WBh2.flatten(), 50)
        axs[1].set_title('Second hidden layer weights')
        axs[2].hist(self.WBo.flatten(), 20)
        axs[2].set_title('Output layer weights')
        plt.tight_layout()
        plt.show()
#----------------------------------------------------------------        
import pickle
with open("dict.txt", "rb") as file:   # Unpickling
    parameters = pickle.load(file)
    
best_model = Network(**parameters)

WBh1 = np.load("WBh1.npy")
WBh2 = np.load("WBh2.npy")
WBo = np.load("WBo.npy")

best_model.load_weights(WBh1,WBh2,WBo)
Z_test = np.loadtxt(fname = 'test_set.txt', delimiter=',')
x_test, y_test = Z_test.T
print("MSE on test_set.txt: %.5f"%best_model.evaluate_mean_loss(x_test,y_test, save=True))
