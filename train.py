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
    Add args.entropy, args.value
    Add CONV4_Net
    Add CONV5_Net
    Add CONV6_Net
    Add second environment and its worker
"""

from __future__ import division
from setproctitle import setproctitle as ptitle
import numpy as np
import torch
import torch.optim as optim
from environment import create_env
from utils import ensure_shared_grads
from model import *  # change to import any models
from player_util import Agent
from torch.autograd import Variable
import gym


def train(rank, args, shared_model, optimizer):
    ptitle('Training Agent: {}'.format(rank))
    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
    torch.manual_seed(args.seed + rank)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + rank)

    # add second environment
    if rank >= args.workers:
        print ('training agent of second environment', rank)
        env = create_env(args.env2, args)
    else:
        env = create_env(args.env, args)
    
    if optimizer is None:
        if args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    env.seed(args.seed + rank)
    player = Agent(None, env, args, None)
    player.gpu_id = gpu_id
    if args.model == 'MLP':
        player.model = A3C_MLP(
            player.env.observation_space.shape[0], player.env.action_space, args.stack_frames)
    if args.model == 'CONV':
        player.model = A3C_CONV(args.stack_frames, player.env.action_space)
    if args.model == 'CONV3':
        player.model = CONV3_Net(args.stack_frames, player.env.action_space)
    if args.model == 'CONV4':
        player.model = CONV4_Net(args.stack_frames, player.env.action_space)
    if args.model == 'CONV5':
        player.model = CONV5_Net(args.stack_frames, player.env.action_space)
    if args.model == 'CONV6':
        player.model = CONV6_Net(args.stack_frames, player.env.action_space)

    player.state = player.env.reset()
    player.state = torch.from_numpy(player.state).float()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.state = player.state.cuda()
            player.model = player.model.cuda()

    # 
    ratio_entropy =args.entropy
    ratio_value = args.value

    player.model.train()   # Sets the module in training mode.
    while True:
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                player.model.load_state_dict(shared_model.state_dict())
        else:
            player.model.load_state_dict(shared_model.state_dict())
        if player.done:
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.cx = Variable(torch.zeros(1, 128).cuda())
                    player.hx = Variable(torch.zeros(1, 128).cuda())
                    if args.model == 'CONV5':
                        player.fx1 = Variable(torch.zeros(4).cuda())
                    elif args.model == 'CONV6':
                        player.cx2 = Variable(torch.zeros(1, 128).cuda())
                        player.hx2 = Variable(torch.zeros(1, 128).cuda())

            else:
                player.cx = Variable(torch.zeros(1, 128))
                player.hx = Variable(torch.zeros(1, 128))
                if args.model == 'CONV5':
                    player.fx1 = Variable(torch.zeros(4))
                elif args.model == 'CONV6':
                    player.cx2 = Variable(torch.zeros(1, 128))
                    player.hx2 = Variable(torch.zeros(1, 128))

        else:
            player.cx = Variable(player.cx.data)
            player.hx = Variable(player.hx.data)
            if args.model == 'CONV5':
                player.fx1 = Variable(player.fx1.data)
            elif args.model == 'CONV6':
                player.cx2 = Variable(player.cx2.data)
                player.hx2 = Variable(player.hx2.data)
           
        for step in range(args.num_steps):

            player.action_train()  # call action_train

            if player.done:
                break

        if player.done:
            player.eps_len = 0
            state = player.env.reset()
            player.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()

        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                R = torch.zeros(1, 1).cuda()
        else:
            R = torch.zeros(1, 1)
        if not player.done:
            state = player.state
            if args.model == 'CONV' or args.model == 'CONV3' or args.model == 'CONV4' or args.model == 'CONV5' or args.model == 'CONV6':
                state = state.unsqueeze(0)
           
            # value is critic
            if args.model == 'CONV5':
                value, _, _, _ = player.model(
                    (Variable(state), (player.hx, player.cx, player.fx1)))
            elif args.model == 'CONV6':
                value, _, _, _ = player.model(
                    (Variable(state), (player.hx, player.cx, player.hx2, player.cx2)))
            else:
                value, _, _, _ = player.model(
                    (Variable(state), (player.hx, player.cx)))
            R = value.data

        player.values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        R = Variable(R)
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                gae = torch.zeros(1, 1).cuda()
        else:
            gae = torch.zeros(1, 1)
        for i in reversed(range(len(player.rewards))):
            R = args.gamma * R + player.rewards[i]
            advantage = R - player.values[i]
            
            # Value Loss is ...
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            # print(player.rewards[i])
            # rewards + gamma* value[i+1]  - value[i]
            delta_t = player.rewards[i] + args.gamma * \
                player.values[i + 1].data - player.values[i].data

            gae = gae * args.gamma * args.tau + delta_t
            
            # Policy Loss is ....
            policy_loss = policy_loss - \
                (player.log_probs[i].sum() * Variable(gae)) - \
                (ratio_entropy * player.entropies[i].sum())

        player.model.zero_grad()

        # --- backward ---
        (policy_loss + ratio_value * value_loss).backward()
        ensure_shared_grads(player.model, shared_model, gpu=gpu_id >= 0)
        optimizer.step()
        player.clear_actions()
