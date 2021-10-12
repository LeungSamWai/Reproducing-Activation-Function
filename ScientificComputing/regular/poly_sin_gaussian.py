#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import argparse
import math
import numpy as np
import numpy.random as npr
from numpy import linalg
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils import data
from torch.utils.data import Dataset
import torch.optim as optim
import time
from math import *
from torch.nn.parameter import Parameter



parser = argparse.ArgumentParser()
parser.add_argument('--dim',type= int, default= 2)
parser.add_argument('--width', type = int, default = 20)
parser.add_argument('--id', type = int, default = 20)
args = parser.parse_args()

dim_set = args.dim
m = args.width
torch.cuda.set_device(args.id)
# In[2]:

class sin_relu2(nn.Module):
    def __init__(self, num_features):
        super(sin_relu2, self).__init__()
        self.width = num_features
        self.q_width = floor(num_features/4)

    def forward(self, x):
        y = torch.zeros(x.size()).cuda()
        y[:,0:self.q_width] = x[:,0:self.q_width]
        y[:,self.q_width:2*self.q_width] = x[:,self.q_width:2*self.q_width]**2
        y[:,2*self.q_width:3*self.q_width] = torch.sin(x[:,2*self.q_width:3*self.q_width])
        y[:,3*self.q_width:self.width] = torch.exp(-x[:,3*self.q_width:self.width]**2/(2*0.05**2))
        return y

class ResNet(nn.Module):
    def __init__(self,dim, m,o):
        super(ResNet, self).__init__()
        self.Ix = torch.zeros([dim,m]).cuda()
        for i in range(dim):
            self.Ix[i,i] = 1
    
        self.fc1 = nn.Linear(dim, m)
        self.fc2 = nn.Linear(m, m)
        
        self.fc3 = nn.Linear(m, m)
        self.fc4 = nn.Linear(m, m)


        
        self.outlayer = nn.Linear(m, o)
        self.relu11 = sin_relu2(m)
        self.relu12 = sin_relu2(m)
        self.relu21 = sin_relu2(m)
        self.relu22 = sin_relu2(m)

    def forward(self, x):
        s = x@self.Ix
        y = self.fc1(x)
        y = self.relu11(y)
        y = self.fc2(y)
        y = self.relu12(y)
        y = y+s
        
        s=y
        y = self.fc3(y)
        y = self.relu21(y)
        y = self.fc4(y)
        y = self.relu22(y)
        y = y+s

                
        output = self.outlayer(y)
        return output
deg = 2

# In[3]:


torch.set_default_tensor_type('torch.DoubleTensor')


# In[4]:


def sol_exact(x):
    N_u = x[:,0:1]**2*(1-x[:,0:1])*x[:,1:2]**2*(1-x[:,1:2])
    return N_u

def right(x):
    f_tem = 2*(1-x[:,0:1])*x[:,0:1]**2*(1-x[:,1:2]) - 4* (1 - x[:,0:1])*x[:,0:1]**2*x[:,1:2] +  2*(1 - x[:,0:1])*(1 - x[:,1:2])*x[:,1:2]**2- 4*x[:,0:1]*(1 - x[:,1:2])*x[:,1:2]**2
    return f_tem 
    


# In[6]:


Batch_size = 10000
def DGM():
    x = torch.rand(Batch_size,2).cuda()
    x.requires_grad = True
    u = model_u(x)
    v= torch.ones(u.shape).cuda()
    ux = torch.autograd.grad(u,x,grad_outputs=v,create_graph=True)[0]
    uxx = torch.zeros(Batch_size,dim_set).cuda()
    for i in range(dim_set):
        ux_tem = ux[:,i].reshape([x.size()[0],1])
        uxx_tem = torch.autograd.grad(ux_tem,x,grad_outputs=v,create_graph=True)[0]
        uxx[:,i] = uxx_tem[:,i]
    loss = torch.sum((-torch.sum(uxx,dim=1).reshape([x.size()[0],1]) + right(x))**2)/Batch_size
    return loss


# In[7]:


model_u1 = ResNet(dim_set,m,1)
device=torch.device("cuda:"+str(args.id) )
model_u1.to(device)


# In[8]:


def model_u(x):
    return torch.prod(x*(1-x),dim=1).reshape([x.size()[0],1])*model_u1(x) 


# In[9]:


def loss_error():
    notes_error = torch.rand(10000,dim_set).cuda()
    notes_error.requires_grad= True
    exact = sol_exact(notes_error)
    u_L2 = sqrt(torch.sum(exact**2)/10000)
    predict= model_u(notes_error)
    value = sqrt(torch.sum((predict[:,0] - exact[:,0])**2)/10000)/u_L2
    return value



    
traintime = 100000
error_save=np.zeros(traintime)
optimizer = optim.Adam([
                {'params': model_u1.parameters()}
            ],lr =1e-3,betas=(0.9, 0.999), eps=1e-8,weight_decay= 1e-4)
time_start=time.time()
from torch.optim.lr_scheduler import StepLR
scheduler =  StepLR(optimizer, step_size=2000, gamma=0.7)
for i in range(traintime):
    optimizer.zero_grad()
    losses = DGM()
    losses.backward()
    optimizer.step() 
    error = loss_error()
    error_save[i]=float(error)

 
    scheduler.step()
    if i%500==1:
        print("i= ",i)
        print("error =",error)
        print("loss1 =",losses.detach())
        np.save("error_save_La_"+str(dim_set)+"D_w"+str(m)+"_poly_sin_gaussian.npy", error_save)
np.save("error_save_La_"+str(dim_set)+"D_w"+str(m)+"_poly_sin_gaussian.npy", error_save)
time_end=time.time()
print('time cost',time_end-time_start,'s')


# In[11]:


import matplotlib.pyplot as plt

plt.plot(np.log10(error_save))


# In[ ]:




