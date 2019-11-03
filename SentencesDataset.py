from torch.utils.data import Dataset
import torch


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
            data.append([self.w2v.wv.vocab[word].index for word in sentence if word in self.w2v.wv.vocab])
        label = self.labels[idx]
        return torch.tensor(data, dtype=torch.long), torch.tensor(label, dtype=torch.long)