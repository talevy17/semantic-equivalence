import pandas as pd
from SentencesDataset import SentencesDataset
from Tests import word2vec


class DataUtils():
    @staticmethod
    def load_data(filename):
        return pd.read_csv(filename, sep="\t", header=None, names=["score", "sant1", "sant2"])

    @staticmethod
    def data2pickle(filename, path_to_save):
        pd.read_csv(filename, sep="\t", header=None, names=["score", "sant1", "sant2"]).to_pickle(path_to_save)

    @staticmethod
    def process_data(data_frame):
        scores = []
        sants = []
        for i in range(len(data_frame.sant1)):
            sant1 = data_frame.sant1[i]
            sant2 = data_frame.sant2[i]
            score = data_frame.score[i]
            if type(sant1) is not float and type(sant2) is not float:
                sants1_word = sant1.split(" ")
                sants2_word = sant2.split(" ")
                sants.append((sants1_word, sants2_word))
                scores.append(score)
        return sants, scores

    @staticmethod
    def load_process_data(filename):
        return DataUtils.process_data(DataUtils.load_data(filename))

    @staticmethod
    def load_dataset(filename, w2v):
        sants, scores = DataUtils.load_process_data(filename)
        return SentencesDataset(sants, scores, w2v)


filename = "C:\\Users\\noamc\\Downloads\\sentences.txt"
dataset = DataUtils.load_dataset(filename, word2vec)
print(type(dataset[0][0]))
