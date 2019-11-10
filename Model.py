import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
import numpy as np
import logging
import pandas as pd
# import nltk
from gensim.models import Word2Vec
import torch
import time
import torch.optim as optim
from DataUtils import DataUtils

num_of_epochs = 5
dropout = 0.25
embedded_dim = 300
hidden_layer = 256
empty_word = torch.from_numpy(np.zeros(embedded_dim, dtype=torch.long))


class Siamese(nn.Module):

    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights,
                 freeze_embeddings=False):
        super(Siamese, self).__init__()
        # remove the cnn

        # self.conv = nn.Sequential(
        #     nn.Conv2d(1, 64, 10),  # 64@96*96
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2),  # 64@48*48
        #     nn.Conv2d(64, 128, 7),
        #     nn.ReLU(),    # 128@42*42
        #     nn.MaxPool2d(2),   # 128@21*21
        #     nn.Conv2d(128, 128, 4),
        #     nn.ReLU(), # 128@18*18
        #     nn.MaxPool2d(2), # 128@9*9
        #     nn.Conv2d(128, 256, 4),
        #     nn.ReLU(),   # 256@6*6
        # )
        # TODO: add lstm or a simple rnn or whatever
        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        self.word_embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(weights), freeze=freeze_embeddings)
        self.lstm = nn.LSTM(embedding_length, hidden_size)
        self.label = nn.Linear(hidden_size, output_size)
        self.liner = nn.Sequential(nn.Linear(9216, 4096), nn.Sigmoid())
        self.out = nn.Linear(4096, 5)

    def lstm_forword(self, input_sentence, batch_size=None):
        pred = self.word_embeddings(
            input_sentence)  # embedded input of shape = (batch_size, num_sequences,  embedding_length)
        pred = pred.permute(1, 0, 2)  # input.size() = (num_sequences, batch_size, embedding_length)
        if batch_size is None:
            h_0 = torch.zeros(1, self.batch_size, self.hidden_size)  # Initial hidden state of the LSTM
            c_0 = torch.zeros(1, self.batch_size, self.hidden_size)  # Initial cell state of the LSTM
        else:
            h_0 = torch.zeros(1, batch_size, self.hidden_size)
            c_0 = torch.zeros(1, batch_size, self.hidden_size)
        output, (final_hidden_state, final_cell_state) = self.lstm(pred, (h_0, c_0))
        final_output = self.label(final_hidden_state[-1])
        return final_output

    def forward_one(self, x):
        x = self.lstm_forword(x)
        x = self.liner(x)
        return x

    def split_input(self, sentences):
        index = (sentences == empty_word).nonzero()
        return sentences[: index], sentences[index + 1:]

    def forward(self, sents):
        x1, x2 = self.split_input(sents)
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        dis = torch.abs(out1 - out2)
        out = self.out(dis)
        return F.softmax(out)
        # return out


def get_accuracy(prediction, y):
    probs = torch.softmax(prediction, dim=1)
    winners = probs.argmax(dim=1)
    correct = (winners == y.argmax(dim=1)).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def train(model, loader, optimizer, criterion, epoch):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    print(f'Epoch: {epoch + 1:02} | Starting Training...')
    for index, batch in enumerate(loader):
        optimizer.zero_grad()
        predictions = model(batch[0]).squeeze(1)
        loss = criterion(predictions, batch[1])
        acc = get_accuracy(predictions, batch[1])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    print(f'Epoch: {epoch + 1:02} | Finished Training')
    return epoch_loss / len(loader), epoch_acc / len(loader)


def time_for_epoch(start, end):
    end_to_end = end - start
    minutes = int(end_to_end / 60)
    seconds = int(end_to_end - (minutes * 60))
    return minutes, seconds


def evaluate(model, loader, criterion, epoch):
    epoch_loss = 0
    epoch_acc = 0
    print(f'Epoch: {epoch + 1:02} | Starting Evaluation...')
    model.eval()
    with torch.no_grad():
        for index, batch in enumerate(loader):
            predictions = model(batch[0]).squeeze(1)
            loss = criterion(predictions, batch[1])
            acc = get_accuracy(predictions, batch[1])
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    print(f'Epoch: {epoch + 1:02} | Finished Evaluation')
    return epoch_loss / len(loader), epoch_acc / len(loader)


def iterate_model(model, train_loader, val_loader):
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MultiLabelSoftMarginLoss()
    for epoch in range(num_of_epochs):
        start_time = time.time()
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, epoch)
        val_loss, val_acc = evaluate(model, val_loader, criterion, epoch)
        end_time = time.time()
        epoch_mins, epoch_secs = time_for_epoch(start_time, end_time)
        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {val_loss:.3f} |  Val. Acc: {val_acc * 100:.2f}%')


def load_w2v():
    return Word2Vec.load("word2vec.model")


def main(filename, train_size):
    w2v = load_w2v()
    dataset = DataUtils.load_dataset(filename, w2v)
    train_len = len(dataset) * train_size
    test_len = len(dataset) - train_len
    train_set, test_set = random_split(dataset, [train_len, test_len])
    net_model = Siamese(batch_size=1, output_size=5, hidden_size=hidden_layer,
                        vocab_size=len(w2v.wv.vocab), embedding_length=embedded_dim, weights=w2v.wv.vectors)
    train_dataloader = DataLoader(train_set, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_set,batch_size=1,shuffle=True)
    iterate_model(net_model, train_dataloader, test_dataloader)


if __name__ == "__main__":
    filename = "C:\\Users\\noamc\\Downloads\\sentences.txt"
    train_size = 0.8
    main(filename,train_size)