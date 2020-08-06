import pickle
import torch
import numpy 
import spacy
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
import os.path

from preprocess import clean

# load spacy tokenizer
nlp = spacy.load('en_core_web_sm',disable=['parser', 'tagger', 'ner'])

class TextDataset(Dataset):
    def __init__(self, raw_csv, vocab_file, cleansing = False):
        # Load vocabulary wrapper
        with open(vocab_file, 'rb') as f:
            vocab = pickle.load(f)
            self.word2idx = vocab['word2idx']
            self.idx2word = vocab['idx2word']

        # raw data from csv file
        raw_data = pd.read_csv(raw_csv, error_bad_lines=False)
        self.sentiment = raw_data.sentiment
        raw_data.loc[raw_data.sentiment=='positive', 'sentiment'] = 1
        raw_data.loc[raw_data.sentiment=='negative', 'sentiment'] = 0

        if cleansing:
            tqdm.pandas(desc='Cleaning empty spaces...') 
            raw_data.review = raw_data.review.progress_apply(lambda x: clean(x))

        tqdm.pandas(desc='Removing empty string...')
        self.sentence_text = raw_data.review.progress_apply(lambda x: x.strip())        
        
        # Load or generate temporary word index
        tqdm.pandas(desc='Converting word to index...')
        if os.path.isfile('sentence_idx_temp.pkl'):
            with open('sentence_idx_temp.pkl', 'rb') as f:
                self.sentence_idx = pickle.load(f)
        else:
            self.sentence_idx = raw_data.review.progress_apply(self.indexer)
            with open('sentence_idx_temp.pkl', 'wb') as f:
                pickle.dump(self.sentence_idx, f)
        
        
    def __len__(self):
        return len(self.sentence_text)
    
    def indexer(self, s):
        index = []
        index.append(1) # <start>
        for w in nlp(s):
            w = w.text.lower()
            if not w in self.word2idx:
                w = '<unk>'
            index.append(self.word2idx[w])
        index.append(2) # <end>
        return index

    def __getitem__(self, idx):
        return torch.Tensor(self.sentence_idx[idx]), self.sentiment[idx]

def collate_fn(data):
    
    data.sort(key=lambda x: len(x[0]), reverse=True)
    sentence_source, sentiment = zip(*data)
    lengths = [len(document) for document in sentence_source]
    sentence = torch.zeros(len(sentence_source), max(lengths)).long()
    
    for i, cap in enumerate(sentence_source):
        end = lengths[i]
        sentence[i, :end] = cap[:end]  

    return sentence, torch.LongTensor(sentiment), lengths

def data_loader(raw_csv, vocab_file, validation_split, batch_size = 64, shuffle = True):

    dataset = TextDataset(raw_csv=raw_csv, vocab_file=vocab_file)
    val_size    = int(validation_split * len(dataset))
    train_size  = len(dataset) - val_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, collate_fn=collate_fn)
    val_loader   = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=1, collate_fn=collate_fn)

    return train_loader, val_loader
