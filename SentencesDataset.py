from torch.utils.data import Dataset
import torch
import numpy as np


embedded_dim = 300

def encoder(w2v, sentence):
    return [w2v.wv.vocab[word].index for word in sentence if word in w2v.wv.vocab]
    # return torch.tensor(encoded, dtype=torch.long)


class SentencesDataset(Dataset):
    def __init__(self, sentences, labels, w2v):


        self.texts = sentences
        self.labels = labels
        self.w2v = w2v

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        dim = len(text[0][0])
        # print(encoder(self.w2v, text[0]).shape)
        # print(encoder(self.w2v, text[1]).shape)
        # print(np.zeros(dim).reshape(1, dim))
        data = []

        sat1 = encoder(self.w2v, text[0])
        sat2 = encoder(self.w2v, text[1])

        x = len(sat1)
        data.append(x)
        data.extend(sat1.copy())
        data.extend(sat2.copy())

        #data = np.concatenate(data) # np.concatenate((sat1 ,np.zeros(dim).reshape(1, dim), sat2), axis=0)
        label = self.labels[idx]
        return torch.tensor(data), torch.tensor(label, dtype=torch.long)
