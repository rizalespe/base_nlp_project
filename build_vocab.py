import pandas as pd
import spacy
from collections import Counter
from tqdm import tqdm
import pickle
from preprocess import clean

threshold = 10
cleansing = False
# load spacy tokenizer
nlp = spacy.load('en_core_web_sm',disable=['parser', 'tagger', 'ner'])

# load the dataset
# sentiment-analysis-dataset.csv
df = pd.read_csv('imdb_dataset.csv', error_bad_lines=False)

if cleansing:
    tqdm.pandas(desc='String cleansing...') 
    df.SentimentText = df.review.progress_apply(lambda x: clean(x))

tqdm.pandas(desc='Removing empty string...') 
sentence_corpus = df.review.progress_apply(lambda x: x.strip())

words = Counter()
tqdm.pandas(desc='Index generating...') 
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

