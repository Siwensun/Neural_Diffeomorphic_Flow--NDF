#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch.nn as nn
import torch
from .module import SdfDecoder
# from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint

class ODEFunc(nn.Module):
    '''
    This refers to the dynamics function f(x,t) in a IVP defined as dh(x,t)/dt = f(x,t). 
    For a given location (t) on point (x) trajectory, it returns the direction of 'flow'.
    Refer to Section 3 (Dynamics Equation) in the paper for details. 
    '''
    def __init__(self, hidden_size, latent_size):
        '''
        Initialization. 
        num_hidden: number of nodes in a hidden layer
        latent_len: size of the latent code being used
        '''
        
        super(ODEFunc, self).__init__()
        
        self.l1 = nn.Linear(3, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)   
        # self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, 3)
        
        self.cond = nn.Linear(latent_size, hidden_size) 

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        
        self.nfe = 0
        self.zeros = torch.zeros((1, latent_size))
        self.latent_dyn_zeros=None
        
    def forward(self, t, cxyz):
        '''
        t: Torch tensor of shape (1,) 
        cxyz: Torch tensor of shape (N, zdim+3). Along dimension 1, the point and shape embeddings are concatenated. 
        
        **NOTE**
        For the uniqueness property to hold, a single dynamics function (operating in 3D) must be used to compute 
        trajectories pertaining to points of a single shape. 
        
        Here, the shape encoding (same for all points of a shape) is used to choose a function which is applied over all the shape points.
        Hence, even though the input xz appears to be a 3+zdim dimensional state, the ODE is still restricted to a 3D state-space. 
        The concatenation is purely to make programming simpler without affecting the underlying theory. 
        
        '''
        point_features = self.relu(self.l1((cxyz[...,-3:]))) # Extract point features #ptsx3 -> #ptsx512
        shape_features = self.tanh(self.cond(cxyz[...,:-3]))  # Extract shape features #ptsxzdim -> #ptsx512
        
        point_shape_features = point_features*shape_features  # Compute point-shape features by elementwise multiplication
        # [Insight :]  Conditioning is critical to allow for several shapes getting learned by same NeuralODE. 
        #              Note that under current formulation, all points belonging to a shape share a common dynamics function.
        
        # Two residual blocks
        point_shape_features = self.relu(self.l2(point_shape_features)) + point_shape_features
        # point_shape_features = self.relu(self.l3(point_shape_features)) + point_shape_features
        # [Insight :] Using less residual blocks leads to drop in performance
        #             while more residual blocks make model heavy and training slow due to more complex trajectories being learned.
        
        dyns_x_t = self.tanh(self.l4(point_shape_features)) #Computed dynamics of point x at time t
        # [Insight :] We specifically choose a tanh activation to get maximum expressivity as observed by He, et.al and Massaroli, et.al
        
        self.nfe+=1  #To check #ode evaluations
        
        # To prevent updating of latent codes during ODESolver calls, we simply make their dynamics all zeros. 
        if self.latent_dyn_zeros is None or self.latent_dyn_zeros.shape[0] != dyns_x_t.shape[0]:
            self.latent_dyn_zeros = self.zeros.repeat(dyns_x_t.shape[0], 1).type_as(dyns_x_t)  
        
        return torch.cat([self.latent_dyn_zeros, dyns_x_t], dim=1) # output is therefore like [0,0..,0, dyn_x, dyn_y, dyn_z] for a point

class NODEBlock(nn.Module):
    '''
    Function to solve an IVP defined as dh(x,t)/dt = f(x,t). 
    We use the differentiable ODE Solver by Chen et.al used in their NeuralODE paper.
    '''
    def __init__(self, odefunc, tol):
        '''
        Initialization. 
        odefunc: The dynamics function to be used for solving IVP
        tol: tolerance of the ODESolver
        '''
        super(NODEBlock, self).__init__()
        self.odefunc = odefunc
        self.cost = 0
        self.rtol = tol
        self.atol = tol

    def define_time_steps(self, end_time, steps, invert):
        times = torch.linspace(0, end_time, steps+1)
        if invert:
            times = times.flip(0)
        return times
        
    def forward(self, cxyz, end_time, steps, invert):
        '''
        Solves the ODE in the forward / reverse time. 
        '''
        self.odefunc.nfe = 0  #To check #ode evaluations
        
        self.times = self.define_time_steps(end_time, steps, invert).to(cxyz)  # Time of integration (must be monotinically increasing!)
        # Solve the ODE with initial condition x and interval time.
        out = odeint(self.odefunc, cxyz, self.times, rtol = self.rtol, atol = self.rtol)
        self.cost = self.odefunc.nfe  # Number of evaluations it took to solve it
        return out


class Warper(nn.Module):
    '''
    A single DeformBlock is made up of two NODE Blocks. Refer secion 3 (Overall Architecture)
    '''
    def __init__(self, 
                 latent_size,
                 hidden_size, 
                 steps,
                 time=1.0,
                 tol = 1e-5):
        super(Warper, self).__init__()
        '''
        Initialization.
        time: some number 0-1
        num_hidden: Number of hidden nodes in the MLP of dynamics
        latent_len: Length of shape embeddings
        tol: tolerance of the ODE Solver
        '''
        
        # Several NODE Blocks
        for step in range(steps):
            setattr(self, f'node{step+1}', NODEBlock(ODEFunc(hidden_size, latent_size), tol = tol))
        
        self.time = time
        self.steps = steps
        
    def forward(self, input=None, invert=False, time=None):
        '''
        Forward/Baclward flow method
        
        input = [code, x]
        x: Nx3 input tensor
        code: Nxzdim tensor embedding
        time: some number 0-1
        
        y: Nx3 output tensor
        '''

        # xyz = input[:, -3:]
        # code = input[:, :-3]

        if time is None:
            time=self.time

        # Note: To enable condioned flows, we concatenate points with their corresponding shape embeddings. 
        #       Refer to code comments in ODEFunc.forward() for more details about this choice.

        time_interval = self.time / (self.steps)

        if not invert:
            xyzs = []
            cxyz = input
            for step in range(self.steps):
                cxyz = getattr(self, f'node{step+1}')(cxyz, time_interval, 1, False)[-1]

                if (step+1) % (max(self.steps // 4, 1)) == 0:
                    xyzs.append(cxyz[:, -3:])
        else:
            xyzs = []
            cxyz = input
            for step in range(self.steps):
                cxyz = getattr(self, f'node{self.steps-step}')(cxyz, time_interval, 1, True)[-1]

                if (step+1) % (max(self.steps // 4, 1)) == 0:
                    xyzs.append(cxyz[:, -3:])
        
        xyz = cxyz[:, -3:]  # output the corresponding 'flown' points.

        return xyz, xyzs

    def timeflow(self, input=None, sub_steps=1):

        # Note: To enable condioned flows, we concatenate points with their corresponding shape embeddings. 
        #       Refer to code comments in ODEFunc.forward() for more details about this choice.

        time_interval = self.time / (self.steps)

        xyzs = []
        cxyz = input
        for step in range(self.steps):
            cxyzs = getattr(self, f'node{self.steps-step}')(cxyz, time_interval, sub_steps, True)[1:]
            cxyz = cxyzs[-1]

            if (step+1) % (max(self.steps // 4, 1)) == 0:
                xyzs.append(cxyzs[:, :, -3:])

        return xyzs


class Decoder(nn.Module):
    def __init__(self, latent_size, warper_kargs, decoder_kargs):
        super(Decoder, self).__init__()
        self.warper = Warper(latent_size, **warper_kargs)
        self.sdf_decoder = SdfDecoder(**decoder_kargs)

    def forward(self, input, invert=False, output_warped_points=False):
        p_final, warped_xyzs = self.warper(input, invert)

        if not self.training:
            x = self.sdf_decoder(p_final)
            if output_warped_points:
                return p_final, x
            else:
                return x
        else:   # training mode, output intermediate positions and their corresponding sdf prediction
            xs = []
            for p in warped_xyzs:
                xs.append(self.sdf_decoder(p))
            if output_warped_points:
                return warped_xyzs, xs
            else:
                return xs

    def forward_template(self, input):
        return self.sdf_decoder(input)
