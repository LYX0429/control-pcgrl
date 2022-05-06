
import numpy as np
import random
import torch
import copy
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import MSELoss
from torch import Tensor

class TinyModel(torch.nn.Module):

    def __init__(self, n, m):
        super(TinyModel, self).__init__()

        self.linear1 = torch.nn.Linear(n * m + 3, 256)
        #self.linear1 = torch.nn.Linear(2 + 3, 256)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(256, 288)
        self.linear3 = torch.nn.Linear(288, n * m)
        #self.linear3 = torch.nn.Linear(288, 2)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        x = self.activation(x)
        return x
        
class ICM():
    
    def __init__(self, game, path):
        if game == 'binary':
            n = 16
            m = 16
        self.weight = 1
        self.reward_type = "both"
        self.path = path
        self.model = TinyModel(n, m)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(next(self.model.parameters()).is_cuda)

        self.criterion = MSELoss()
        self.optimizer = SGD(self.model.parameters(), lr=0.01, momentum=0.9)
    
    def predict(self, state, action, next_state):
        x = state.flatten()
        y = next_state.flatten()
        x = np.append(x, action)
        x = np.array(x)
        y = np.array(y)
        
        x = Tensor(x)
        y = Tensor(y)
        x = x.to(self.device)
        y = y.to(self.device)

        for epoch in range(11):
            self.optimizer.zero_grad()
            yhat = self.model(x)
            loss = self.criterion(yhat, y)
            loss.backward()
            self.optimizer.step()
        
        reward = loss.item()
        return reward * self.weight
    
    def decode(self, ob):
        obs = [[0 for j in range(len(ob[0]))] for i in range(len(ob))]
        for i in range(len(ob)):
            for j in range(len(ob[0])):
                obs[i][j] = np.argmax(ob[i][j])
        return np.array(obs)
        
    def save(self):
        torch.save(self.model.state_dict(), self.path)

if __name__ == '__main__':

    n = 16
    m = 16
    
    icm = ICM('binary', './')
    # icm.model.load_state_dict(torch.load("binary_ctrl-narrow-v0_both_30.model", map_location=torch.device('cpu')))
    
    dataSet = []
    actionSet = []
    valSet = []
    num = 10000
    for x in range(num):
        data = []
        for i in range(n):
            for j in range(m):
                data.append(random.randint(0, 1))
                
        action = [random.randint(0, 15), random.randint(0, 15), random.randint(0, 2)]
        
        val = copy.copy(data)
        if action[2] == 1:
            val[action[1] * 16 + action[0]] = 0
        elif action[2] == 2:
            val[action[1] * 16 + action[0]] = 1
            
        actionSet.append(action)
        dataSet.append(data)
        valSet.append(val)
    dataSet = np.array(dataSet)
    actionSet = np.array(actionSet)
    valSet = np.array(valSet)
    
    for i in range(num):
        reward = icm.predict(dataSet[i], np.array([1,2,3]), valSet[i])
        print(reward)

