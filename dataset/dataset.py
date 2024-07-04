import ast
import csv
import os.path
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.rotation import matrix_to_rotation_6d, quaternion_to_matrix


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
        left_path = os.path.join(dataset_path, 'left.csv')
        right_path = os.path.join(dataset_path, 'right.csv')
        # read all the data, loop
        left_action = read_tracker_data(left_path)
        right_action = read_tracker_data(right_path)
        action_clip, cond_clips, label_clips = self.segment_action(left_action, right_action)
        total_act.extend(action_clip)
        total_cond.extend(cond_clips)
        total_lebel.extend(label_clips)
        ###
        self.action = torch.from_numpy(np.array(total_act))
        self.cond_act = torch.from_numpy(np.array(total_cond))
        self.label = torch.from_numpy(np.array(total_lebel).astype(np.float32))

        return
    def __len__(self):
        return len(self.action)

    def segment_action(self, action_left, action_right, label=None):
        pos_left = action_left['pos'].astype(np.float32)  # (T, 3)
        pos_right = action_right['pos'].astype(np.float32)  # (T, 3)
        assert pos_left.shape[0] == pos_right.shape[0]
        pos = np.concatenate((pos_left, pos_right), axis=1)  # (T, 6)

        quat_left = torch.from_numpy(action_left['quat'].astype(np.float32))  # (T, 4)
        rot6d_left = matrix_to_rotation_6d(quaternion_to_matrix(quat_left)).numpy()  # (T, 6)
        stat_left = action_left['stat'].astype(np.float32)  # (T, 1)
        quat_right = torch.from_numpy(action_right['quat'].astype(np.float32))
        rot6d_right = matrix_to_rotation_6d(quaternion_to_matrix(quat_right)).numpy()  # (T, 6)
        stat_right = action_right['stat'].astype(np.float32)
        assert pos_left.shape[0] == quat_left.shape[0] == stat_left.shape[0]

        rot6d = np.concatenate((rot6d_left, rot6d_right), axis=1)  # (T, 8)
        stat = np.concatenate((np.expand_dims(stat_left, axis=1), np.expand_dims(stat_right, axis=1)), axis=1)  # (T, 2)
        act = np.concatenate((pos, rot6d, stat), axis=1)  # (T, 16)
        padding = act[0:1, ...].repeat(self.cond_length-1, 0)   # (cond_length-1, 16)

        padded_act = np.concatenate((padding, act), axis=0)
        total_length = padded_act.shape[0]  # (T+cond_length-1, 16)

        actions = []
        labels = []
        cond_acts = []

        for i in range(0, total_length-self.clip_length-self.cond_length+1):
            temp_cond = padded_act[i:i+self.cond_length, ...]
            temp_action = padded_act[i+self.cond_length:i+self.cond_length+self.clip_length, ...]
            if i is not total_length-self.clip_length-self.cond_length:
                stop_token = np.zeros_like(padded_act[0:1, ...])
                temp_action = np.concatenate((temp_action, stop_token), axis=0)
            else:
                stop_token = np.ones_like(padded_act[0:1, ...])
                temp_action = np.concatenate((temp_action, stop_token), axis=0)
            # temp_label = label
            temp_label = 1
            actions.append(temp_action)
            cond_acts.append(temp_cond)
            labels.append(temp_label)

        return actions, cond_acts, labels

    def __getitem__(self, item):
        action = self.action[item]
        cond_act = self.cond_act[item]
        label = self.label[item]
        return action, cond_act, label

if __name__ == '__main__':
    datapath = '../data/try/pant_blackjean/left.csv'
    pose = read_tracker_data(datapath)