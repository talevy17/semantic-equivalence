from torch.utils.data import Dataset
import torch
import numpy as np


def encoder(w2v, sentence):
    return np.asarray([w2v.wv.vocab[word].index for word in sentence if word in w2v.wv.vocab])
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
        data = np.concatenate((encoder(self.w2v, text[0]), encoder(self.w2v, text[1])), axis=0)
        label = self.labels[idx]
        return torch.from_numpy(data), torch.tensor(label, dtype=torch.long)
