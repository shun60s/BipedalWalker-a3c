# -*- coding: utf-8 -*-

"""
-----------------------------------------------------------------------------
   Copyright 2017 David Griffis

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
-----------------------------------------------------------------------------
Changed:
            add LSTM_USE flag in A3C_MLP

"""


from __future__ import division
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from utils import norm_col_init, weights_init, weights_init_mlp


class A3C_CONV(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super(A3C_CONV, self).__init__()
        # nn.Conv1d(in_channels, out_channels, kernel_size, strid
        self.conv1 = nn.Conv1d(num_inputs, 32, 3, stride=1, padding=1)
        self.lrelu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv1d(32, 32, 3, stride=1, padding=1)
        self.lrelu2 = nn.LeakyReLU(0.1)
        self.conv3 = nn.Conv1d(32, 64, 2, stride=1, padding=1)
        self.lrelu3 = nn.LeakyReLU(0.1)
        self.conv4 = nn.Conv1d(64, 64, 1, stride=1)
        self.lrelu4 = nn.LeakyReLU(0.1)


        """
              nn.Conv1d(in_channels, out_channels, kernel_size, strid
        torch.nn.LSTMCell(input_size: int, hidden_size(=output_size): int, bias: bool = True)

        """
        self.lstm = nn.LSTMCell(1600, 128)
        num_outputs = action_space.shape[0]
        self.critic_linear = nn.Linear(128, 1)
        self.actor_linear = nn.Linear(128, num_outputs)
        self.actor_linear2 = nn.Linear(128, num_outputs)

        self.apply(weights_init)
        lrelu_gain = nn.init.calculate_gain('leaky_relu')
        self.conv1.weight.data.mul_(lrelu_gain)
        self.conv2.weight.data.mul_(lrelu_gain)
        self.conv3.weight.data.mul_(lrelu_gain)
        self.conv4.weight.data.mul_(lrelu_gain)

        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.actor_linear2.weight.data = norm_col_init(
            self.actor_linear2.weight.data, 0.01)
        self.actor_linear2.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()

    def forward(self, inputs):
        x, (hx, cx) = inputs

        x = self.lrelu1(self.conv1(x))
        x = self.lrelu2(self.conv2(x))
        x = self.lrelu3(self.conv3(x))
        x = self.lrelu4(self.conv4(x))

        x = x.view(x.size(0), -1)  # Auto Size Adjust like reshape
        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        return self.critic_linear(x), F.softsign(self.actor_linear(x)), self.actor_linear2(x), (hx, cx)



class A3C_MLP(torch.nn.Module):
    def __init__(self, num_inputs, action_space, n_frames):
        super(A3C_MLP, self).__init__()
        # Applies a linear transformation to the incoming data: y=xAT+b
        self.fc1 = nn.Linear(num_inputs, 256)
        self.lrelu1 = nn.LeakyReLU(0.1) # negative_slope 0.1
        self.fc2 = nn.Linear(256, 256)
        self.lrelu2 = nn.LeakyReLU(0.1)
        self.fc3 = nn.Linear(256, 128)
        self.lrelu3 = nn.LeakyReLU(0.1)
        self.fc4 = nn.Linear(128, 128)
        self.lrelu4 = nn.LeakyReLU(0.1)

        self.LSTM_USE=True  # add. Set False only when n_frames is 1

        if self.LSTM_USE:
            self.m1 = n_frames * 128  # n_frames : stack_frame of gym
            self.lstm = nn.LSTMCell(self.m1, 128)
        
        # critic is output only onr value
        # actior uses two linear and output number is action_space.shape[0]
        num_outputs = action_space.shape[0]
        self.critic_linear = nn.Linear(128, 1)
        self.actor_linear = nn.Linear(128, num_outputs)
        self.actor_linear2 = nn.Linear(128, num_outputs)

        self.apply(weights_init_mlp)
        lrelu = nn.init.calculate_gain('leaky_relu')
        self.fc1.weight.data.mul_(lrelu)
        self.fc2.weight.data.mul_(lrelu)
        self.fc3.weight.data.mul_(lrelu)
        self.fc4.weight.data.mul_(lrelu)

        # init weights
        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.actor_linear2.weight.data = norm_col_init(
            self.actor_linear2.weight.data, 0.01)
        self.actor_linear2.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        
        if self.LSTM_USE:
            self.lstm.bias_ih.data.fill_(0)
            self.lstm.bias_hh.data.fill_(0)

        self.train()  # Sets the module in training mode.

    def forward(self, inputs):
        x, (hx, cx) = inputs

        x = self.lrelu1(self.fc1(x))
        x = self.lrelu2(self.fc2(x))
        x = self.lrelu3(self.fc3(x))
        x = self.lrelu4(self.fc4(x))

        if self.LSTM_USE:  # add for only MLP use
            x = x.view(1, self.m1)           # input [n_frames, 128]
            hx, cx = self.lstm(x, (hx, cx))  # output 128
            x = hx

        return self.critic_linear(x), F.softsign(self.actor_linear(x)), self.actor_linear2(x), (hx, cx)
