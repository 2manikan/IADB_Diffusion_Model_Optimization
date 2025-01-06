# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 21:25:27 2025

@author: Meenakshi Manikandan
"""

import torch
import matplotlib.pyplot as plt



#plotting the original code's loss curve (AdamW optimizer) vs. the modified code's loss curve (SGD optimizer)
num_points=200
offset=0

loss_orig=torch.load("losses99800")[offset:num_points+offset]
plt.plot(range(offset,len(loss_orig)+offset) , loss_orig, color='black')


losses=torch.load("ls99800")[offset:num_points+offset]
plt.plot(range(offset,len(losses)+offset), losses, color='red')
plt.show()