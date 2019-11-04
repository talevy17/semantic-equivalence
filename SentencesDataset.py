from torch.utils.data import Dataset
import torch
import numpy as np


class SentencesDataset(Dataset):
    def __init__(self, sentences, labels, w2v):
        self.texts = sentences
        self.labels = labels
        self.w2v = w2v

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        data = []
        for sentence in text:
            data.append(np.array([self.w2v.wv.vocab[word].index for word in sentence if word in self.w2v.wv.vocab]))
        label = self.labels[idx]
        x = torch.tensor(data[0], dtype=torch.long),torch.tensor(data[1], dtype=torch.long)
        x = np.ndarray.tolist(x)
        return torch.tensor(x), torch.tensor(label, dtype=torch.long)