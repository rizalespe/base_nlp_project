import pandas as pd
import spacy
from collections import Counter
from tqdm import tqdm
import pickle

threshold = 10

tqdm.pandas(desc='Progress')
# load spacy tokenizer
nlp = spacy.load('en_core_web_sm',disable=['parser', 'tagger', 'ner'])

# load the dataset
df = pd.read_csv('sentiment-analysis-dataset.csv', error_bad_lines=False) 
sentence_corpus = df.SentimentText.progress_apply(lambda x: x.strip())

words = Counter()

for sentence in tqdm(sentence_corpus.values):
    words.update(w.text.lower() for w in nlp(sentence))

original_length = len(words)
# discard token less then threshold
words = [word for word, cnt in words.items() if cnt >= threshold]

# add <pad> and <unk> token to vocab which will be used later
words = ['<pad>','<start>','<end>','<unk>'] + words

# create word to index dictionary and reverse
word2idx = {o:i for i,o in enumerate(words)}
idx2word = {i:o for i,o in enumerate(words)}

print("Total token: ",len(word2idx),"/",original_length)
vocab = {}
vocab['word2idx'] = word2idx
vocab['idx2word'] = idx2word

with open('vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)

