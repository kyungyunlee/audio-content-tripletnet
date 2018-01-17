import os
import torch
from torch.utils.data import Dataset
import numpy as np

class MSD_Dataset(Dataset):
    def __init__(self, mel_dir, triplets_path):
        self.triplets = np.loadtxt(triplets_path, dtype='str', delimiter=',')
        self.mel_dir = mel_dir


    def __getitem__(self, index):
        triplet = self.triplets[index]
        
        try:
            pos1 = np.load(self.mel_dir + triplet[0] + '.npy')
            pos2 = np.load(self.mel_dir + triplet[1] + '.npy')
            neg = np.load(self.mel_dir + triplet[2] + '.npy')
        except :
            print("Data cannot be loaded")
            return None

        entry = {'pos1' : pos1, 'pos2' : pos2, 'neg' : neg}
        return entry

    def __len__(self):
        return self.triplets.shape[0]





