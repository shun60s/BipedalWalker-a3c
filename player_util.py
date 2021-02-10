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
        Add CONV3_Net

"""

from __future__ import division
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from utils import normal  # , pi


class Agent(object):
    def __init__(self, model, env, args, state):
        self.model = model
        self.env = env
        self.state = state
        self.hx = None
        self.cx = None
        self.eps_len = 0
        self.args = args
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.done = True
        self.info = None
        self.reward = 0
        self.gpu_id = -1

    def action_train(self):
        if self.args.model == 'CONV' or self.args.model == 'CONV3':
            self.state = self.state.unsqueeze(0)
        # value: critic
        # mu: softsign liner out　action
        # sigma: liner out action
        # hx lstm
        # cx lstm
        value, mu, sigma, (self.hx, self.cx) = self.model(
            (Variable(self.state), (self.hx, self.cx)))
        mu = torch.clamp(mu, -1.0, 1.0)
        # SoftPlus is a smooth approximation to the ReLU function and can be used to constrain the output of a machine 
        # to always be positive.
        # 1/beta * log(1+exp(beta*x))
        sigma = F.softplus(sigma) + 1e-5

        # Returns a tensor filled with random numbers from a normal distribution with mean 0 and variance 1 
        # (also called the standard normal distribution).
        # The shape of the tensor is defined by the variable argument size
        eps = torch.randn(mu.size())


        pi = np.array([math.pi])
        pi = torch.from_numpy(pi).float()
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                eps = Variable(eps).cuda()
                pi = Variable(pi).cuda()
        else:
            eps = Variable(eps)
            pi = Variable(pi)

        action = (mu + sigma.sqrt() * eps).data
        act = Variable(action)
        
        # prob
        #     normal
        #      a = (-1 * (x - mu).pow(2) / (2 * sigma)).exp()
        #      b = 1 / (2 * sigma * pi.expand_as(sigma)).sqrt()
        #     return a * b
        prob = normal(act, mu, sigma, self.gpu_id, gpu=self.gpu_id >= 0)
        action = torch.clamp(action, -1.0, 1.0)

        # entropy 
        # Expand this tensor to the same size as other . 
        # self.expand_as(other) is equivalent to self.expand(other.size()) 
        entropy = 0.5 * ((sigma * 2 * pi.expand_as(sigma)).log() + 1)
        
        self.entropies.append(entropy)
        log_prob = (prob + 1e-6).log()
        self.log_probs.append(log_prob)
        state, reward, self.done, self.info = self.env.step(
            action.cpu().numpy()[0])
        reward = max(min(float(reward), 1.0), -1.0)
        self.state = torch.from_numpy(state).float()
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()
        self.eps_len += 1
        self.done = self.done or self.eps_len >= self.args.max_episode_length
        self.values.append(value)
        self.rewards.append(reward)
        return self

    def action_test(self):
        with torch.no_grad():  # without save parameters, only forward
            if self.done:
                if self.gpu_id >= 0:
                    with torch.cuda.device(self.gpu_id):
                        self.cx = Variable(torch.zeros(
                            1, 128).cuda())
                        self.hx = Variable(torch.zeros(
                            1, 128).cuda())
                else:
                    self.cx = Variable(torch.zeros(1, 128))
                    self.hx = Variable(torch.zeros(1, 128))
            else:
                self.cx = Variable(self.cx.data)
                self.hx = Variable(self.hx.data)
            if self.args.model == 'CONV' or self.args.model == 'CONV3':
                self.state = self.state.unsqueeze(0)  #### reshape(mini-batch,...) for picture input
            value, mu, sigma, (self.hx, self.cx) = self.model(
                (Variable(self.state), (self.hx, self.cx)))

        #Clamp all elements in input into the range [ min, max ] and return a resulting tensor:
        mu = torch.clamp(mu.data, -1.0, 1.0)

        # When test, action is directory to use mu,  vs when train prob...
        #
        action = mu.cpu().numpy()[0]
        state, self.reward, self.done, self.info = self.env.step(action)
        self.state = torch.from_numpy(state).float()
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()
        self.eps_len += 1
        self.done = self.done or self.eps_len >= self.args.max_episode_length
        return self

    def clear_actions(self):
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        return self
