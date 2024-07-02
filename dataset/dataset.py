import os
import math
import random

from tqdm import tqdm
import ast
import csv
import numpy as np
import torch
import codecs as cs
from itertools import islice
from torch.utils.data import sampler, Dataset, DataLoader, RandomSampler

def read_tracker_data(file_path):
    """Read the tracker data from a file and return the position and quaternion."""
    positions = []
    quaternions = []
    grippers =[]
    action = {}
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            # if i % 20 == 0:  # Process every 30th row
            pos = ast.literal_eval(row['position'])
            ori = ast.literal_eval(row['orientation'])
            positions.append(np.array([pos['x'], pos['y'], pos['z']]))
            quaternions.append([
                round(ori['w'], 3),
                round(ori['x'], 3),
                round(ori['y'], 3),
                round(ori['z'], 3)])
            grippers.append(int(row['gripper']))
    quat = np.array(quaternions)
    stat = np.array(grippers)
    action['pos'] = np.array(positions)
    action['quat'] = np.array(quaternions)
    action['stat'] = np.array(grippers)
    return action

def read_tracker(filename, with_key=True):
    labels = []
    returnArray = []
    lines = open(filename).readlines()
    if with_key: # skip first line
        labels = lines[0].strip().split(',')
        lines = lines[1:]
    for line in lines:
        line = line.strip().split(',"')
        print(line[2])

    return

class Action_Dataset(Dataset):
    # This dataset is the NeMF latent sequences dataset with text descriptions
    def __init__(self, dataset_path, args, **kwargs):
        self.clip_length = args.clip_length
        self.cond_length = args.cond_length
        total_act = []
        total_cond = []
        total_lebel = []
        # read all the data, loop
        left_action = read_tracker()
        right_action = read_tracker()
        action_clip, cond_clips, label_clips = self.segment_action(left_action, right_action)
        total_act.extend(action_clip)
        total_cond.extend(cond_clips)
        total_lebel.extend(label_clips)

        self.action = torch.from_numpy(total_act).to.torch.float32()
        self.cond_act = torch.from_numpy(total_cond).to.torch.float32()
        self.label = total_lebel

        return
    def __len__(self):
        return

    def segment_action(self, action_left, action_right, label=None):
        pos_left = action_left['pos']
        pos_right = action_right['pos']
        assert pos_left.shape[0] == pos_right.shape[0]
        pos = np.concatenate((pos_left, pos_right), axis=1)

        quat_left = action_left['quat']
        stat_left = action_left['stat']

        quat_right = action_right['quat']
        stat_right = action_right['stat']
        assert pos_left.shape[0] == quat_left.shape[0] == stat_left.shape[0]

        quat = np.concatenate((quat_left, quat_right), axis=1)
        stat = np.concatenate((stat_left, stat_right), axis=1)
        act= np.concatenate((pos, quat, stat), axis=1)
        padding = act[0, ...].repeat(self.cond_length-1, 1)
        padded_act = np.concatenate((padding, act), axis=0)
        total_length = len(padded_act.shape[0])

        actions = []
        labels = []
        cond_acts = []

        for i in range(0, total_length-self.clip_length-self.cond_length+1):
            temp_cond = padded_act[i:i+self.cond_length, ...]
            temp_action = padded_act[i+self.cond_length:i+self.cond_length+self.clip_length, ...]
            temp_label = label
            actions.append(temp_action)
            cond_acts.append(temp_cond)
            label.append(temp_label)

        return actions, cond_acts, labels

    def __getitem__(self, item):
        action = self.action[item]
        cond_act = self.cond_act[item]
        label = self.label[item]
        return action, cond_act, label

if __name__ == '__main__':
    datapath = '../data/try/pant_blackjean/left.csv'
    pose = read_tracker_data(datapath)