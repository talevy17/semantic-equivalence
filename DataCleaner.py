

class DataCleaner:
    @staticmethod
    def parse_sentece_to_words(sentence, to_ignore):
        sent = []
        for word in sentence:
            if not word in to_ignore:
                sent += word
        return sent