import torch.utils.data as data
import pandas as pd
from utilities import *
from torchnlp.encoders.text import StaticTokenizerEncoder
from sklearn.model_selection import train_test_split
from collections import Counter

from transformers import BertTokenizer
import torch
from eda_nlp.code.eda import eda


def create_dataset(df, author_idx=None, multiclass=None):
    """
    creates a balanced dataset, where the author_idx is mapped to the label 1
    and the negatives (label 0) are randomly sampled from all other authors
    The dataframe is expected to be sorted by date in a descending order
    :param undersampling:
    :param df:
    """
    assert multiclass or author_idx is not None

    # convert the dataframe to lists of samples
    X = df['Sentence'].values.tolist()
    y = df['Label'].values.tolist()

    # split the data into train, val and test set
    X, X_test, y, y_test = train_test_split(X, y, stratify=y, test_size=0.05, shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.1, shuffle=True)

    # after having a stratified split we can adjust the labels to correctly sample them afterwards
    if author_idx is not None:
        y_train = [1 if label == author_idx else 0 for label in y_train]
        y_val = [1 if label == author_idx else 0 for label in y_val]
        y_test = [1 if label == author_idx else 0 for label in y_test]

        print("Samples amount in training set:")
        print("Author: {}, Not Author: {}".format(y_train.count(1), y_train.count(0)))
        print("Samples amount in validation:")
        print("Author: {}, Not Author: {}".format(y_val.count(1), y_val.count(0)))
        print("Samples amount in test set:")
        print("Author: {}, Not Author: {}".format(y_test.count(1), y_test.count(0)))
    else:
        print("Samples amount in training set:")
        print(Counter(y_train))
        print("Samples amount in validation:")
        print(Counter(y_val))
        print("Samples amount in test set:")
        print(Counter(y_test))

    return X_train, y_train, X_val, y_val, X_test, y_test


def preprocess_sentences_dataframe(dataframe_path,
                                   drop_short_sentences=False,
                                   remove_contractions=True,
                                   convert_to_lowercase=True,
                                   multiclass=None,
                                   single_class_author_idx=None,
                                   min_occurences=1,
                                   remove_stopwords=False,
                                   processing='punct',
                                   split_data=True,
                                   remove_trump_sentences=True):
    """
    :param dataframe_path:
    :param drop_unnecessary:
    :param drop_short_sentences:
    :param remove_contractions:
    :param convert_to_lowercase:
    :param min_occurences: #word to be included in the vocabulary
    :return: embedded/encoded sentences, labels
    """

    assert multiclass or single_class_author_idx is not None
    # ----- Pre-process the dataframe ----- #
    print("--- Creating the dataset ----")
    df = pd.read_pickle(dataframe_path)
    punct_remover = RegexpTokenizer(r'\w+')

    # to lowercase
    if convert_to_lowercase:
        df['Sentence'] = df['Sentence'].apply(str.lower)

    # remove contractions, since they can't usually be handled correctly
    if remove_contractions:
        df['Sentence'] = df['Sentence'].apply(remove_contractions_from_sentence)

    if remove_stopwords:
        df['Sentence'] = df['Sentence'].apply(remove_stopwords_from_sentence)

    # drop sentences with less than 3 words
    if drop_short_sentences:
        df = df[df['Sentence'].map(word_counter) > 2]
        df.reset_index(drop=True, inplace=True)

    if remove_trump_sentences:
        # since we got way to much data from Donald Trump, which would lead to immense over-/undersampling,
        # we remove the oldest bits of his data (which are the last entries in the dataframe, since its sorted)
        trump_sentences = df[df['Label'] == 5].index.tolist()
        amount_trump_sentences = len(trump_sentences)
        amount_trump_sentences_to_remove = int(0.8 * amount_trump_sentences)
        trump_sentences_to_remove = trump_sentences[amount_trump_sentences - amount_trump_sentences_to_remove:]
        df = df.drop(trump_sentences_to_remove)
        df.reset_index(inplace=True, drop=True)

    # ----- Create the Datasets ----- #
    if split_data:
        X_train, y_train, X_val, y_val, X_test, y_test = create_dataset(df, author_idx=single_class_author_idx, multiclass=multiclass)
    else:
        return df['Sentence'].values.tolist()

    # ----- Tokenize Sentences ----- #
    if 'bert' in processing:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        X_train = tokenizer(X_train, truncation=True, padding=True)
        X_val = tokenizer(X_val, truncation=True, padding=True)
        X_test = tokenizer(X_test, truncation=True, padding=True)

    elif 'svm' in processing:
        # concatenate training and validation set, since SVMs dont need three datasets
        X = []
        X.extend(X_train)
        X.extend(X_val)
        X_train = X
        y = []
        y.extend(y_train)
        y.extend(y_val)
        y_train = y

    else:
        sentences = []
        sentences.extend(X_train)
        sentences.extend(X_val)
        # create a lookup table for the whole vocabulary except the test data
        lookup_table = StaticTokenizerEncoder(sentences, min_occurrences=min_occurences, tokenize=lambda s: punct_remover.tokenize(s))

        max_sentence_length = max([len(punct_remover.tokenize(s)) for s in X_train])

    if 'bert' in processing:
        return X_train, y_train, X_val, y_val, X_test, y_test
    elif 'svm' in processing:
        return X_train, y_train, X_test, y_test
    else:
        return X_train, y_train, X_val, y_val, X_test, y_test, lookup_table, lookup_table.vocab_size, max_sentence_length


def augment_dataset_with_EDA(X, y, classes_w_stronger_augmentation=[]):

    X_augmented = []
    y_augmented = []
    for i in range(len(y)):
        # augment the data once for each augmentation technique
        try:
            if y[i] in classes_w_stronger_augmentation:
                augmented = eda(X[i], alpha_ri=0.05, alpha_rs=0.05, alpha_sr=0.05, p_rd=0.05, num_aug=12)[:-1]
            else:
                augmented = eda(X[i], num_aug=4)[:-1]
            # add the original sentence
            augmented.append(X[i])
            # add the (unchanged) labels
            labels = [y[i]] * len(augmented)
            X_augmented.extend(augmented)
            y_augmented.extend(labels)
        except ValueError:
            X_augmented.append(X[i])
            y_augmented.append(y[i])

    return X_augmented, y_augmented


class SentenceDataset(data.Dataset):

    def __init__(self, sentences, labels, lookup_table, sentence_size, pre_padding=False, packing=False, shuffle_wordorder=False):

        self.labels = labels
        self.sentences = sentences
        self.lookup_table = lookup_table
        self.sentence_size = sentence_size
        self.pre_padding = pre_padding
        self.packing = packing
        self.shuffle_wordorder = shuffle_wordorder

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]

        # convert the sentence into indices
        sentence = self.lookup_table.encode(sentence).long()

        # if the word is unknown to the lookup table we want to treat it as padding
        # since we didn't train an embedding for it, so we should leave it out
        # all unknown words would be mapped to the same <unk> token, which would confuse the classifier
        sentence[sentence==1] = 0

        if self.shuffle_wordorder:
            # if we have a tensor with label 1 we randomly shuffle it and reverse the label with a certain probability
            if label == 1:
                if torch.rand(1).item() < 0.3:
                    sentence = sentence[torch.randperm(sentence.size()[0])]
                    label = 0

        s_length = len(sentence)

        # if the sentence is shorter than the desired sentence length, we pad it
        if s_length < self.sentence_size:
            sentence = pad_sentence(sentence, self.sentence_size, self.pre_padding)

        # if the sentence is longer, we cut off the extending parts
        else:
            sentence = sentence[:self.sentence_size]
            s_length = len(sentence)

        if self.packing:
            return sentence, label, s_length
        else:
            return sentence, label


class LM_Dataset(data.Dataset):

    def __init__(self, sentences, labels):

        self.labels = labels
        self.sentences = sentences

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.sentences.items()}
        item['label'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class LM_DatasetWithTokenizer(data.Dataset):

    def __init__(self, sentences, labels, tokenizer, augment_data):

        self.labels = labels
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.augment_data = augment_data

    def __getitem__(self, idx):
        s = self.sentences[idx]
        if self.augment_data:
            s = string_augmentator(0.6, 0.6, s)
        s = self.tokenizer(s, truncation=True, padding='max_length')
        item = {key: torch.tensor(val) for key, val in s.items()}
        item['label'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)