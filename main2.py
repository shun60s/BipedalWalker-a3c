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
    Add args.save last
    Add args.entropy, args.value
    Add CONV3_Net
    Add CONV4_Net
    Add CONV5_Net
    Add args.last_load
    Add CONV6_Net
    Add second environment and its worker
"""


from __future__ import print_function, division
import os
os.environ["OMP_NUM_THREADS"] = "1" # numpy...をimportする前に使えるスレッド数を制限
import argparse
import torch
import torch.multiprocessing as mp
from environment import create_env
from model import *  # change to import any models
from train import train
from test import test
from shared_optim import SharedRMSprop, SharedAdam
import time


parser = argparse.ArgumentParser(description='A3C')
parser.add_argument(
    '--lr',
    type=float,
    default=0.0001,
    metavar='LR',
    help='learning rate (default: 0.0001)')
parser.add_argument(
    '--gamma',
    type=float,
    default=0.99,
    metavar='G',
    help='discount factor for rewards (default: 0.99)')
parser.add_argument(
    '--tau',
    type=float,
    default=1.00,
    metavar='T',
    help='parameter for GAE (default: 1.00)')
parser.add_argument(
    '--seed',
    type=int,
    default=1,
    metavar='S',
    help='random seed (default: 1)')
parser.add_argument(
    '--workers',
    type=int,
    default=32,
    metavar='W',
    help='how many training processes to use (default: 32)')
parser.add_argument(
    '--workers2',
    type=int,
    default=0,
    metavar='W2',
    help='how many training processes to use for second environment(default: 0 off)')
parser.add_argument(
    '--num-steps',
    type=int,
    default=20,
    metavar='NS',
    help='number of forward steps in A3C (default: 300)')
parser.add_argument(
    '--max-episode-length',
    type=int,
    default=10000,
    metavar='M',
    help='maximum length of an episode (default: 10000)')
parser.add_argument(
    '--env',
    default='BipedalWalker-v2',
    metavar='ENV',
    help='environment to train on (default: BipedalWalker-v2)')
parser.add_argument(
    '--env2',
    default='BipedalWalkerHardcoreStump1-v0',
    metavar='ENV2',
    help='second environment to train on (default: BipedalWalkerHardcoreStump1-v0)')
parser.add_argument(
    '--shared-optimizer',
    default=True,
    metavar='SO',
    help='use an optimizer without shared statistics.')
parser.add_argument(
    '--load',
    default=False,
    metavar='L',
    help='load a trained model')
parser.add_argument(
    '--load-last',
    default=False,
    metavar='LL',
    help='load model dict and optimizer dicts when shared_optimizer.')
parser.add_argument(
    '--save-max',
    default=True,
    metavar='SM',
    help='Save model on every test run high score matched or bested')
parser.add_argument(
    '--save-last',
    default=False,
    metavar='SL',
    help='Save last model on every test')
parser.add_argument(
    '--optimizer',
    default='Adam',
    metavar='OPT',
    help='shares optimizer choice of Adam or RMSprop')
parser.add_argument(
    '--load-model-dir',
    default='trained_models/',
    metavar='LMD',
    help='folder to load trained models from')
parser.add_argument(
    '--save-model-dir',
    default='trained_models/',
    metavar='SMD',
    help='folder to save trained models')
parser.add_argument(
    '--log-dir',
    default='logs/',
    metavar='LG',
    help='folder to save logs')
parser.add_argument(
    '--model',
    default='MLP',
    metavar='M',
    help='Model type to use')
parser.add_argument(
    '--stack-frames',
    type=int,
    default=1,
    metavar='SF',
    help='Choose number of observations to stack')
parser.add_argument(
    '--gpu-ids',
    type=int,
    default=-1,
    nargs='+',
    help='GPUs to use [-1 CPU only] (default: -1)')
parser.add_argument(
    '--amsgrad',
    default=True,
    metavar='AM',
    help='Adam optimizer amsgrad parameter')
parser.add_argument(
    '--entropy',
    type=float,
    default=0.01,
    metavar='EN',
    help='entropy rate (default: 0.01)')
parser.add_argument(
    '--value',
    type=float,
    default=0.5,
    metavar='VA',
    help='value rate in Loss function (default: 0.5)')
parser.add_argument(
    '--initweight',
    default=False,
    metavar='IW',
    help='initialize CONV5 weights using previous CONV4 dat')

# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
# Implemented multiprocessing using locks but was not beneficial. Hogwild
# training was far superior

if __name__ == '__main__':
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if args.gpu_ids == -1:
        args.gpu_ids = [-1]
    else:
        torch.cuda.manual_seed(args.seed)
        mp.set_start_method('spawn')
    env = create_env(args.env, args)
    if args.model == 'MLP':
        shared_model = A3C_MLP(
            env.observation_space.shape[0], env.action_space, args.stack_frames)
    if args.model == 'CONV':
        shared_model = A3C_CONV(args.stack_frames, env.action_space)
    if args.model == 'CONV3':
        shared_model = CONV3_Net(args.stack_frames, env.action_space)
    if args.model == 'CONV4':
        shared_model = CONV4_Net(args.stack_frames, env.action_space)
    if args.model == 'CONV5':
        shared_model = CONV5_Net(args.stack_frames, env.action_space)
    if args.model == 'CONV6':
        shared_model = CONV6_Net(args.stack_frames, env.action_space)

    if args.load:
        saved_state = torch.load('{0}{1}.dat'.format(
            args.load_model_dir, args.env), map_location=lambda storage, loc: storage)
        shared_model.load_state_dict(saved_state)
    
    # choice one of args.load or args.load_last
    if args.load_last:
        saved_state = torch.load('{0}{1}_last.dat'.format(
            args.load_model_dir, args.env), map_location=lambda storage, loc: storage)
        shared_model.load_state_dict(saved_state)

    #----
    # for initialize CONV5 weights using previous CONV4 dat
    if args.initweight:
        param1 = torch.load('trained_models/BipedalWalkerHardcore-v2_CONV4_Net.dat')
        param2 = shared_model.state_dict()
        param2['h4fc1.weight'] = param1['h4fc1.weight']
        param2['h4fc2.weight'] = param1['h4fc2.weight']
        param2['j01fc1.weight'] = param1['j01fc1.weight']
        param2['j01fc2.weight'] = param1['j01fc2.weight']
        param2['j23fc1.weight'] = param1['j23fc1.weight']
        param2['j23fc2.weight'] = param1['j23fc2.weight']
        param2['g02fc1.weight'] = param1['g02fc1.weight']
        param2['g02fc2.weight'] = param1['g02fc2.weight']
        param2['fc3.weight'] = param1['fc3.weight']
        #param2['alfc1.weight'] = param1['alfc1.weight']
        #param2['alfc2.weight'] = param1['alfc2.weight']
        param2['critic_linear.weight'] = param1['critic_linear.weight']
        param2['actor_linear.weight'] = param1['actor_linear.weight']
        param2['actor_linear2.weight'] = param1['actor_linear2.weight']
     
        param2['conv1.weight'] = param1['conv1.weight']
        param2['conv2.weight'] = param1['conv2.weight']
        param2['conv3.weight'] = param1['conv3.weight']
        param2['conv4.weight'] = param1['conv4.weight']
    
        param2['h4fc1.bias'] = param1['h4fc1.bias']
        param2['h4fc2.bias'] = param1['h4fc2.bias']
        param2['j01fc1.bias'] = param1['j01fc1.bias']
        param2['j01fc2.bias'] = param1['j01fc2.bias']
        param2['j23fc1.bias'] = param1['j23fc1.bias']
        param2['j23fc2.bias'] = param1['j23fc2.bias']
        param2['g02fc1.bias'] = param1['g02fc1.bias']
        param2['g02fc2.bias'] = param1['g02fc2.bias']
        param2['fc3.bias'] = param1['fc3.bias']
        #param2['alfc1.bias'] = param1['alfc1.bias']
        #param2['alfc2.bias'] = param1['alfc2.bias']           
        param2['critic_linear.bias'] = param1['critic_linear.bias']
        param2['actor_linear.bias'] = param1['actor_linear.bias']
        param2['actor_linear2.bias'] = param1['actor_linear2.bias']
        shared_model.load_state_dict(param2)
        
        print ('init weights...')
    #----


    shared_model.share_memory()

    if args.shared_optimizer:
        if args.optimizer == 'RMSprop':
            optimizer = SharedRMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = SharedAdam(
                shared_model.parameters(), lr=args.lr, amsgrad=args.amsgrad)

        if args.load_last:
            saved_state = torch.load('{0}{1}_last_opt.dat'.format(
                args.load_model_dir, args.env), map_location=lambda storage, loc: storage)
            optimizer.load_state_dict(saved_state)

        optimizer.share_memory()
    else:
        optimizer = None

    processes = []

    # testを起動　これが定期的に結果を表示してMAX SAVEする
    p = mp.Process(target=test, args=(args, shared_model, optimizer)) # change
    p.start()
    processes.append(p)
    time.sleep(0.1)

    # trainをworkers個 起動
    for rank in range(0, args.workers + args.workers2):
        p = mp.Process(target=train, args=(
            rank, args, shared_model, optimizer))
        p.start()
        processes.append(p)
        time.sleep(0.1)
    for p in processes:
        time.sleep(0.1)
        p.join()
