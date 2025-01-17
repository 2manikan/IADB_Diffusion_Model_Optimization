# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 17:57:54 2025

@author: Meenakshi Manikandan
"""

from torch.optim import Optimizer
import torch
import time
import math






#goal: create a new state dict storing the parameter values. perform steps on this state dict. then load that into param groups.
class NewOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, weight_decay=0.01, betas=(0.9,0.999)):
        #THIS LINE HERE creates the state_dict. This state dict (get from <optim_instance>.state_dict()) is fundamental to an optimizer
        #This links the parameters to the id's. This is only required for the optimizer here.
        
        
        #bunch of "NONE" values here. that makes sense because this '__init__" function is called much before loss.backward()
        super(NewOptimizer, self).__init__(params, defaults={'lr':lr, 'weight_decay':weight_decay, 'betas':betas})
        
        #create a list of moving averages. for EACH SET OF WEIGHTS.
        self.epsilon=1e-8
        self.num_params=len(self.state_dict()['param_groups'][0]['params'])        
        self.moving_average=[0] * self.num_params #create a moving average for each parameter
        self.multiple_moments=[0] * self.num_params
        self.weight_decay=weight_decay
        
        self.parameters=[]
        params=self.param_groups[0]["params"]
        for i in range(self.num_params):
            self.parameters.append(params[i])
            self.state[params[i]]["step"]=torch.zeros(())
            self.state[params[i]]["first_moment"]=torch.zeros_like(params[i])
            self.state[params[i]]["second_moment"]=torch.zeros_like(params[i])
        
        

        

    #loss.backward will have calculated the gradients for the required tensors
    #update the weight values by ONE STEP based on LEARNING RATE, WEIGHT DECAY, AND GRAD.
        #update the weights in PARAM GROUPS using a FOR LOOP -- NOT FOR EACH (bc this changes the actual stored values not just copies of them)
    #no need to return anything
    def step(self):
        
        for i in range(self.num_params):
            parameter=self.parameters[i]
            first_m=self.state[parameter]['first_moment']
            second_m=self.state[parameter]['second_moment']
            weight_decay=self.state_dict()['param_groups'][0]['weight_decay']
            betas=self.state_dict()['param_groups'][0]['betas']
            lr=self.state_dict()['param_groups'][0]['lr']
            
            
            #step update
            self.state[parameter]['step']=torch.tensor(self.state[parameter]['step'].item()+1)
            step_number=self.state[parameter]['step']
            
            #weight decay
            if(weight_decay!=0):
                parameter.grad.add(parameter, alpha=weight_decay)
            
            
                
            #update moving average
            first_m.lerp_( parameter.grad,(1-betas[0]))
            second_m.mul_(betas[1]).addcmul_(parameter.grad, parameter.grad.conj(), value=1 - betas[1]) 
            
            
            #bias correction and update
            first_bias=1-betas[0] ** step_number
            second_bias= 1-betas[1] **step_number
            step_size=lr/first_bias
            second_bias_sqrt=second_bias ** 0.5
            denominator=(second_m.sqrt() /second_bias_sqrt).add_(self.epsilon)
            parameter.data.addcdiv_(first_m, denominator, value=-step_size)
            
            
            
            
        
        """weight_decay=self.param_groups[0]['weight_decay']
        lr=self.param_groups[0]['lr']
        betas=self.param_groups[0]['betas']
       
        
        
        
        for i in range(self.num_params):
            weight_set=self.param_groups[0]['params'][i]
            
            
            
            #print("---->",weight_set[0][0][0])
           
            #weight decay
            if(weight_decay!=0):
                #print(weight_set.grad.data[0][0][0])
                weight_set.grad.data += weight_decay * weight_set
                #print(weight_decay * weight_set[0][0][0])
                #print(weight_set.grad.data[0][0][0])    
                
            #calculate moving averages
            if weight_set not in self.state:
                self.state[weight_set] = {'first_moment': torch.zeros_like(weight_set), 'second_moment': torch.zeros_like(weight_set)}
                #set timestep
                self.state[weight_set]['timestep']=1
            
            #print("BEFORE: ",self.state[weight_set]['first_moment'][0][0][0])            
            self.state[weight_set]['first_moment'] = betas[0] * self.state[weight_set]['first_moment'] + (1-betas[0]) * weight_set.grad.data
            self.state[weight_set]['second_moment'] = betas[1] * self.state[weight_set]['second_moment'] + (1-betas[1]) * (weight_set.grad.data ** 2)
            
            
            #bias correction
            self.state[weight_set]['first_moment'] = self.state[weight_set]['first_moment']/(1-(betas[0]**self.state[weight_set]['timestep']))
            self.state[weight_set]['second_moment'] = self.state[weight_set]['second_moment']/(1-(betas[1]**self.state[weight_set]['timestep']))
            
            #print("AFTER: ",self.state[weight_set]['first_moment'][0][0][0])
            #print("-----------------------------------")
            
            #update weights
            step_to_make=lr * (self.state[weight_set]['first_moment']/(torch.sqrt(self.state[weight_set]['second_moment']) + self.epsilon))
            #print("Tensor Here: ",step_to_make[0][0][0])
            weight_set.data = weight_set - step_to_make
            
            #update step number
            self.state[weight_set]['timestep']+=1"""
           
            