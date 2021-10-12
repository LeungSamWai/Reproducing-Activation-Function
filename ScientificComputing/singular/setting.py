import torch.nn as nn
import torch
from math import *

class Rational(torch.nn.Module):
    """Rational Activation function.
    It follows:
    `f(x) = P(x) / Q(x),
    where the coefficients of P and Q are initialized to the best rational 
    approximation of degree (3,2) to the ReLU function
    # Reference
        - [Rational neural networks](https://arxiv.org/abs/2004.01902)
    """
    def __init__(self):
        super().__init__()
        self.coeffs = torch.nn.Parameter(torch.Tensor(4, 2))
        self.reset_parameters()

    def reset_parameters(self):
        self.coeffs.data = torch.Tensor([[1.1915, 0.0],
                                    [1.5957, 2.383],
                                    [0.5, 0.0],
                                    [0.0218, 1.0]])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.coeffs.data[0,1].zero_()
        exp = torch.tensor([3., 2., 1., 0.], device=input.device, dtype=input.dtype)
        X = torch.pow(input.unsqueeze(-1), exp)
        PQ = X @ self.coeffs
        output = torch.div(PQ[..., 0], PQ[..., 1])
        return output
    
class poly(nn.Module):
    def __init__(self, num_features):
        super(poly, self).__init__()
        self.width = num_features
        self.q_width = floor(num_features/2)

    def forward(self, x):
        y = torch.zeros(x.size()).cuda()
        y[:,0:self.q_width] = x[:,0:self.q_width]
        y[:,self.q_width:self.width] = x[:,self.q_width:self.width]**2
        return y
    
class poly_relu(nn.Module):
    def __init__(self, num_features):
        super(poly_relu, self).__init__()
        self.width = num_features
        self.q_width = floor(num_features/3)

    def forward(self, x):
        y = torch.zeros(x.size()).cuda()
        y[:,0:self.q_width] = x[:,0:self.q_width]
        y[:,self.q_width:2*self.q_width] = x[:,self.q_width:2*self.q_width]**2
        y[:,2*self.q_width:self.width] = torch.relu(x[:,2*self.q_width:self.width])
        return y
    
class poly_relu3(nn.Module):
    def __init__(self, num_features):
        super(poly_relu3, self).__init__()
        self.width = num_features
        self.q_width = floor(num_features/3)

    def forward(self, x):
        y = torch.zeros(x.size()).cuda()
        y[:,0:self.q_width] = x[:,0:self.q_width]
        y[:,self.q_width:2*self.q_width] = x[:,self.q_width:2*self.q_width]**2
        y[:,2*self.q_width:self.width] = torch.relu(x[:,2*self.q_width:self.width])**3
        return y

class poly_sin(nn.Module):
    def __init__(self, num_features):
        super(poly_sin, self).__init__()
        self.width = num_features
        self.q_width = floor(num_features/3)

    def forward(self, x):
        y = torch.zeros(x.size()).cuda()
        y[:,0:self.q_width] = x[:,0:self.q_width]
        y[:,self.q_width:2*self.q_width] = x[:,self.q_width:2*self.q_width]**2
        y[:,2*self.q_width:self.width] = torch.sin(x[:,2*self.q_width:self.width])
        return y
    
class relu3(nn.Module):
    def __init__(self, num_features):
        super(relu3, self).__init__()

    def forward(self, x):
        y = torch.relu(x)**3
        return y
class poly_sin_gaussian(nn.Module):
    def __init__(self, num_features):
        super(poly_sin_gaussian, self).__init__()
        self.width = num_features
        self.q_width = floor(num_features/4)
    def forward(self, x):
        y = torch.zeros(x.size()).cuda()
        y[:,0:self.q_width] = x[:,0:self.q_width]
        y[:,self.q_width:2*self.q_width] = x[:,self.q_width:2*self.q_width]**2
        y[:,2*self.q_width:3*self.q_width] = torch.sin(x[:,2*self.q_width:3*self.q_width])
        y[:,3*self.q_width:self.width] = torch.exp(-x[:,3*self.q_width:self.width]**2/(2*0.1**2))
        return y   
class poly_sin_gaussian_rational(nn.Module):
    def __init__(self, num_features):
        super(poly_sin_gaussian_rational, self).__init__()
        self.width = num_features
        self.q_width = floor(num_features/5)
        self.rational = Rational()
    def forward(self, x):
        y = torch.zeros(x.size()).cuda()
        y[:,0:self.q_width] = x[:,0:self.q_width]
        y[:,self.q_width:2*self.q_width] = x[:,self.q_width:2*self.q_width]**2
        y[:,2*self.q_width:3*self.q_width] = torch.sin(x[:,2*self.q_width:3*self.q_width])
        y[:,3*self.q_width:4*self.q_width] = torch.exp(-x[:,3*self.q_width:4*self.q_width]**2/(2*0.1**2))
        y[:,4*self.q_width:self.width] = self.rational(x[:,4*self.q_width:self.width])
        return y   
    
class poly_sin_cos_vary(nn.Module):
    def __init__(self, num_features):
        super(poly_sin_cos_vary, self).__init__()
        self.width = num_features
        self.q_width = floor(num_features/3)
        self.qq_width = floor(num_features/6)
        self.range = torch.arange(1,floor(num_features/6)+1).cuda()
    def forward(self, x):
        y = torch.zeros(x.size()).cuda()
        y[:,0:self.q_width] = x[:,0:self.q_width]
        y[:,self.q_width:self.q_width+self.qq_width] = torch.sin(2*pi*self.range*x[:,self.q_width:self.qq_width+self.q_width])
        y[:,self.q_width+self.qq_width:2*self.q_width] = torch.cos(2*pi*self.range*x[:,self.q_width+self.qq_width:2*self.q_width])
        y[:,2*self.q_width:self.width] = x[:,2*self.q_width:self.width]**2
        return y     
    
class poly_sin_vary(nn.Module):
    def __init__(self, num_features):
        super(poly_sin_vary, self).__init__()
        self.width = num_features
        self.q_width = floor(num_features/3)
        self.range = torch.arange(1,floor(num_features/3)+1).cuda()
    def forward(self, x):
        y = torch.zeros(x.size()).cuda()
        y[:,0:self.q_width] = x[:,0:self.q_width]
        y[:,self.q_width:2*self.q_width] = torch.sin(2*pi*self.range*x[:,self.q_width:2*self.q_width])
        y[:,2*self.q_width:self.width] = x[:,2*self.q_width:self.width]**2
        return y  
    
class poly_sin_gaussian_vary(nn.Module):
    def __init__(self, num_features):
        super(poly_sin_gaussian_vary, self).__init__()
        self.width = num_features
        self.q_width = floor(num_features/4)
        self.range = torch.arange(0.05,0.1,1/(floor(num_features/4))).cuda()
    def forward(self, x):
        y = torch.zeros(x.size()).cuda()
        y[:,0:self.q_width] = x[:,0:self.q_width]
        y[:,self.q_width:2*self.q_width] = torch.exp(-x[:,self.q_width:2*self.q_width]**2/(2*self.range**2))
        y[:,2*self.q_width:3*self.q_width] = torch.sin(x[:,2*self.q_width:3*self.q_width])
        y[:,3*self.q_width:self.width] = x[:,3*self.q_width:self.width]**2
        return y 
    
class Rational(torch.nn.Module):
    """Rational Activation function.
    It follows:
    `f(x) = P(x) / Q(x),
    where the coefficients of P and Q are initialized to the best rational 
    approximation of degree (3,2) to the ReLU function
    # Reference
        - [Rational neural networks](https://arxiv.org/abs/2004.01902)
    """
    def __init__(self):
        super().__init__()
        self.coeffs = torch.nn.Parameter(torch.Tensor(4, 2))
        self.reset_parameters()

    def reset_parameters(self):
        self.coeffs.data = torch.Tensor([[1.1915, 0.0],
                                    [1.5957, 2.383],
                                    [0.5, 0.0],
                                    [0.0218, 1.0]])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.coeffs.data[0,1].zero_()
        exp = torch.tensor([3., 2., 1., 0.], device=input.device, dtype=input.dtype)
        X = torch.pow(input.unsqueeze(-1), exp)
        PQ = X @ self.coeffs
        output = torch.div(PQ[..., 0], PQ[..., 1])
        return output

    




    
class ResNet(nn.Module):
    def __init__(self,in_feature, width, out_feature, method):
        super(ResNet, self).__init__()
        self.Ix = torch.zeros([in_feature,width]).cuda()
        for i in range(in_feature):
            self.Ix[i,i] = 1
    
        self.fc1 = nn.Linear(in_feature, width)
        self.fc2 = nn.Linear(width, width)
        
        self.fc3 = nn.Linear(width, width)
        self.fc4 = nn.Linear(width, width)


        
        self.outlayer = nn.Linear(width, out_feature)
        if method == "poly":
            self.relu11 = poly(width)
            self.relu12 = poly(width)
            self.relu21 = poly(width)
            self.relu22 = poly(width)
        elif method == "poly_relu":
            self.relu11 = poly_relu(width)
            self.relu12 = poly_relu(width)
            self.relu21 = poly_relu(width)
            self.relu22 = poly_relu(width)
        elif method == "poly_relu3":
            self.relu11 = poly_relu3(width)
            self.relu12 = poly_relu3(width)
            self.relu21 = poly_relu3(width)
            self.relu22 = poly_relu3(width)
        elif method == "poly_sin":
            self.relu11 = poly_sin(width)
            self.relu12 = poly_sin(width)
            self.relu21 = poly_sin(width)
            self.relu22 = poly_sin(width)
        elif method == "poly_sin_first_vary":
            self.relu11 = poly_sin_vary(width)
            self.relu12 = poly_sin(width)
            self.relu21 = poly_sin(width)
            self.relu22 = poly_sin(width)
        elif method == "poly_sin_last_vary":
            self.relu11 = poly_sin(width)
            self.relu12 = poly_sin(width)
            self.relu21 = poly_sin(width)
            self.relu22 = poly_sin_vary(width)
        elif method == "poly_sin_cos_first_vary":
            self.relu11 = poly_sin_cos_vary(width)
            self.relu12 = poly_sin(width)
            self.relu21 = poly_sin(width)
            self.relu22 = poly_sin(width)
        elif method == "poly_sin_cos_last_vary":
            self.relu11 = poly_sin(width)
            self.relu12 = poly_sin(width)
            self.relu21 = poly_sin(width)
            self.relu22 = poly_sin_cos_vary(width)
        elif method == "relu3":
            self.relu11 = relu3(width)
            self.relu12 = relu3(width)
            self.relu21 = relu3(width)
            self.relu22 = relu3(width)  
            
        elif method == "poly_sin_gaussian":
            self.relu11 = poly_sin_gaussian(width)
            self.relu12 = poly_sin_gaussian(width)
            self.relu21 = poly_sin_gaussian(width)
            self.relu22 = poly_sin_gaussian(width)  
        elif method == "poly_sin_gaussian_vary":
            self.relu11 = poly_sin_gaussian_vary(width)
            self.relu12 = poly_sin_gaussian(width)
            self.relu21 = poly_sin_gaussian(width)
            self.relu22 = poly_sin_gaussian(width) 
        elif method == "poly_sin_gaussian_rational":
            self.relu11 = poly_sin_gaussian_rational(width)
            self.relu12 = poly_sin_gaussian_rational(width)
            self.relu21 = poly_sin_gaussian_rational(width)
            self.relu22 = poly_sin_gaussian_rational(width) 
        elif method == "rational":
            self.relu11 = Rational()
            self.relu12 = Rational()
            self.relu21 = Rational()
            self.relu22 = Rational()
        elif method == "wendland2":
            self.relu11 = wendland0
            self.relu12 = wendland0
            self.relu21 = wendland0
            self.relu22 = wendland0
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