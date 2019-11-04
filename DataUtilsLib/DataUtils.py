import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

from torch.utils import data


class DataUtils():
    @staticmethod
    def load_data(filename):
        return pd.read_csv(filename, sep="\t", header=None, names=["score", "sant1", "sant2"])

    @staticmethod
    def data2pickle(filename, path_to_save):
        pd.read_csv(filename, sep="\t", header=None, names=["score", "sant1", "sant2"]).to_pickle(path_to_save)

    @staticmethod
    def process_data(data_frame):
        scores = data_frame.score
        sants1 = [word.split(" ") for word in data_frame.sant1 if type(word) != float]
        sants2 = [word.split(" ") for word in data_frame.sant2 if type(word) != float]
        sants = []
        for i in range(len(scores)):
            sants.append((sants1, sants2))
        return sants, scores

    @staticmethod
    def load_process_data(filename):
        return DataUtils.process_data(DataUtils.load_data(filename))


class SantsDataset(data.Dataset):
    def __init__(self, santas, scores, transform=None):
        self.sants = santas
        self.scores = scores

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, item):
        score = self.scores[item]
        sants = self.sants[item]
        # TODO: use word embedding to change the wors to vectors
        return sants, torch.tensor(score)  # torch.tensor(sants), torch.tensor(score)

    @staticmethod
    def getDataset(filename):
        sants, scores = DataUtils.load_process_data(filename)
        return SantsDataset(sants, scores)


filename = "C:\\Users\\noamc\\Downloads\\sentences.txt"
test = SantsDataset.getDataset(filename)[0][0]
print("\n\n\n\n\n\n")
print(len(test), end='\n')
print("\n\n\n\n\n\n")
print("first item:", end='\n')
print("\n\n\n\n\n\n")
print(len(test[0]), end='\n')
print("\n\n\n\n\n\n")
print(test[0], end='\n')
print("\n\n\n\n\n\n")
print("second item:", end='\n')
print("\n\n\n\n\n\n")
print(len(test[1]), end='\n')
print("\n\n\n\n\n\n")
print(test[1], end='\n')
print("\n\n\n\n\n\n")
