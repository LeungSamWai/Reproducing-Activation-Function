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
parser.add_argument('--id', type = int, default = 20)
parser.add_argument('--method', type = str)

args = parser.parse_args()
dim_set = args.dim
m = args.width
torch.cuda.set_device(args.id)


# In[3]:


torch.set_default_tensor_type('torch.DoubleTensor')


# In[4]:


def sol_exact(x):
    x_Norm = torch.sqrt(torch.sum(x**2,dim=1).reshape([x.size()[0],1]))
    N_u = torch.sin(2*pi*(1-x_Norm))
    return N_u

def right(x):
    x_Norm = torch.sqrt(torch.sum(x**2,dim=1).reshape([x.size()[0],1]))
    f_tem = - dim_set * 2*pi*torch.cos(2*pi*(1-x_Norm)) - x_Norm*(2*pi)**2* torch.sin(2*pi*(1-x_Norm))
    return f_tem 
    


# In[6]:


Batch_size = 10000
def DGM():
    x = generate_uniform_points_in_sphere(dim_set,1,Batch_size).cuda()
    x.requires_grad = True
    x_Norm = torch.sqrt(torch.sum(x**2,dim=1).reshape([x.size()[0],1]))
    u = model_u(x)
    v= torch.ones(u.shape).cuda()
    ux = torch.autograd.grad(u,x,grad_outputs=v,create_graph=True)[0]
    aux = x_Norm*ux
    uxx = torch.zeros(Batch_size,dim_set).cuda()
    for i in range(dim_set):
        ux_tem = aux[:,i].reshape([x.size()[0],1])
        uxx_tem = torch.autograd.grad(ux_tem,x,grad_outputs=v,create_graph=True)[0]
        uxx[:,i] = uxx_tem[:,i]
    Laplce = torch.sum(uxx,dim=1).reshape([x.size()[0],1])
    f_tem = -Laplce 
    loss = torch.sum((f_tem + right(x))**2)/Batch_size
    return loss


# In[7]:


model_u1 = ResNet(dim_set,m,1,args.method)
device=torch.device("cuda:"+str(args.id) )
model_u1.to(device)


# In[8]:

def model_u(x):
    x_Norm = torch.sqrt(torch.sum(x**2,dim=1).reshape([x.size()[0],1]))
    return (1-x_Norm)*model_u1(x) 


# In[9]:

def generate_uniform_points_in_sphere(d,R,N):
    points = torch.randn(N,d)
    scales = (R*torch.rand(N))**(1/d)
    notes = points/(torch.sqrt(torch.sum(points**2,dim=1))/scales).reshape(N,1)*torch.ones(N ,d)
    return notes

def loss_error():
    notes_error = generate_uniform_points_in_sphere(dim_set,1,Batch_size).cuda()
    notes_error.requires_grad= True
    exact = sol_exact(notes_error)
    u_L2 = sqrt(torch.sum(exact**2)/10000)
    predict= model_u(notes_error)
    value = sqrt(torch.sum((predict[:,0] - exact[:,0])**2)/10000)/u_L2
    return value


def loss_error_inf():
    notes_error = generate_uniform_points_in_sphere(dim_set,1,Batch_size).cuda()
    notes_error[0,:] = torch.tensor([0,0]).cuda()
    exact = sol_exact(notes_error)
    predict= model_u(notes_error)
    value = torch.max(torch.abs(exact-predict))
    return value
    
traintime = 500000
error_save=np.zeros(traintime)
error_save_inf=np.zeros(traintime)
optimizer = optim.Adam([
                {'params': model_u1.parameters()}
            ],lr =1e-3,betas=(0.9, 0.999), eps=1e-8,weight_decay= 1e-4)
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
    error_inf = loss_error_inf()
    error_save_inf[i] = error_inf
 
    scheduler.step()
    if i%500==1:
        print("i= ",i)
        print("error =",error)
        print("error_inf = ", error_inf)
        print("loss1 =",losses.detach())
        np.save("error_save_La_"+str(dim_set)+"D_w"+str(m)+"_"+str(args.method)+".npy", error_save)
np.save("error_save_La_"+str(dim_set)+"D_w"+str(m)+"_"+str(args.method)+".npy", error_save)
time_end=time.time()
print('time cost',time_end-time_start,'s')
torch.save(model_u1.state_dict(),"model_save_la"+str(dim_set)+"D_w"+str(m)+"_"+str(args.method)+".pt")
# In[11]:


import matplotlib.pyplot as plt

plt.plot(np.log10(error_save))


# In[ ]:




