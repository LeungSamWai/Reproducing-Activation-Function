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
from setting import *


parser = argparse.ArgumentParser()
parser.add_argument('--dim',type= int, default= 2)
parser.add_argument('--width', type = int, default = 20)
parser.add_argument('--freq', type = int, default = 20)
parser.add_argument('--id', type = int, default = 20)
parser.add_argument('--method', type = str)

args = parser.parse_args()
dim_set = args.dim
m = args.width
k = args.freq
torch.cuda.set_device(args.id)
import torch
torch.manual_seed(0)

# In[3]:


torch.set_default_tensor_type('torch.DoubleTensor')


# In[4]:


def sol_exact(x):
    N_u = torch.prod(torch.sin(2*k*pi*x),dim=1).reshape([x.size()[0],1])
    return N_u

def right(x):
    N_u = torch.prod(torch.sin(2*k*pi*x),dim=1).reshape([x.size()[0],1])
    f_tem = 2*(2*k)**2*pi**2*N_u -(N_u-1)**3 +(N_u+2)**2
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
    loss = torch.sum((-torch.sum(uxx,dim=1).reshape([x.size()[0],1]) -(u-1)**3 +(u+2)**2 - right(x))**2)/Batch_size
    return loss


# In[7]:


model_u1 = ResNet(dim_set,m,1,args.method)
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
            ],lr =1e-3,betas=(0.9, 0.999), eps=1e-8 )
time_start=time.time()
from torch.optim.lr_scheduler import StepLR
scheduler =  StepLR(optimizer, step_size=1000, gamma=0.95)
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
        np.save("error_save_La_"+str(dim_set)+"D_w"+str(m)+"_k"+str(k)+"_"+str(args.method)+".npy", error_save)
np.save("error_save_La_"+str(dim_set)+"D_w"+str(m)+"_k"+str(k)+"_"+str(args.method)+".npy", error_save)
time_end=time.time()
print('time cost',time_end-time_start,'s')
torch.save(model_u1.state_dict(),"model_save_la"+str(dim_set)+"D_w"+str(m)+"_"+str(args.method)+".pt")
# In[11]:


import matplotlib.pyplot as plt

plt.plot(np.log10(error_save))


# In[ ]:




