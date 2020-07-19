from easydict import EasyDict
import torch
import os

config = EasyDict()

config.gpu = "1"

# dataloader jobs number
config.num_workers = 12

# batch_size
config.batch_size = 2

# training epoch number
config.max_epoch = 200

config.start_epoch = 0

# learning rate
config.lr = 1e-4

# using GPU
config.cuda = True

config.output_dir = 'output'

config.input_size = 640

# max polygon per image
config.max_annotation = 200

# max point per polygon
config.max_points = 20

# use hard examples (annotated as '#')
config.use_hard = True

# prediction on 1/scale feature map
config.scale = 1

# Control coefficient
config.fuc_k = [1,4,7,11]

# demo tcl threshold
config.threshold = 0.3

# min kernal area size (default: 5)
config.min_area = 150

# min confidence value
config.score_i = 0.8

# select the filtering algorithm
config.voting = False


def update_config(config, extra_config):
    for k, v in vars(extra_config).items():
        config[k] = v
    # print(config.gpu)
    config.device = torch.device('cuda') if config.cuda else torch.device('cpu')


def print_config(config):
    print('==========Options============')
    for k, v in config.items():
        print('{}: {}'.format(k, v))
    print('=============End=============')
