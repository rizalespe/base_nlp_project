import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
import wandb
wandb.init(project="nlp_document_classification", group="experiment_1", job_type="training_evaluation")

from custom_csv_dataset import data_loader
from models_v2 import SimpleModel 

'''
Training & validation structure:
# epoch
    # train (actual loss)
        # validation (get the mean loss)
    # test
    
'''

# Setting & Device configuration 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vocab_file = 'vocab.pkl'
epoch = 30
learning_rate = 0.0001
n_output = 1
print_every = 100
validation_split = 0.2
batch_size= 64
shuffle= True
embedding_size= 200
hidden_size=64
num_layers=2
alpha = 0.9

model = "models_v2"
raw_csv = 'imdb_dataset.csv'


with open(vocab_file, 'rb') as f:
    vocab = pickle.load(f)

# Dataset loading
train_loader, val_loader = data_loader( raw_csv = raw_csv, 
                                        vocab_file = 'vocab.pkl', 
                                        validation_split = validation_split, 
                                        batch_size= batch_size, 
                                        shuffle= shuffle)

# Model training
model = SimpleModel(vocab_size=len(vocab['idx2word']), 
                    embedding_size= embedding_size, 
                    hidden_size=hidden_size, 
                    num_layers=num_layers, 
                    n_output = n_output).to(device)

wandb.watch(model)

# Loss and optimizer
criterion = nn.BCELoss()
params = model.parameters()
# optimizer = torch.optim.Adam(params, lr=learning_rate)
optimizer = torch.optim.RMSprop(params, lr = learning_rate, alpha=alpha)

wandb.config.update({   "dataset":raw_csv,
                        "epochs": epoch, 
                        "learning_rate": learning_rate,
                        "n_output":n_output,
                        "print_every":print_every,
                        "validation_split":validation_split,
                        "batch_size": batch_size,
                        "shuffle":shuffle,
                        "embedding_size":embedding_size,
                        "hidden_size":hidden_size,
                        "num_layers":num_layers,
                        "model":model,
                        "alpha":alpha
                        })


total_step = len(train_loader)
for epoch_id in range(epoch):
    
    for idx, (sentence, sentiment, length) in enumerate(train_loader):
        num_correct_train = 0

        # send the data into the GPU device
        sentence, sentiment = sentence.to(device), sentiment.to(device)

        # zero the parameter and gradient
        model.zero_grad()
        
        # forward
        model.train()
        prediction = model(sentence, length)

        # Pengukuran akurasi training
        # convert output probabilities to predicted class (0 or 1)
        pred_train = torch.round(prediction.squeeze())  # rounds to the nearest integer

        # compare predictions to true label
        correct_tensor_train = pred_train.eq(sentiment.float().view_as(pred_train))
        
        if device == 'cpu':
            correct_train = np.squeeze(correct_tensor_train.numpy()) 
        else: 
            correct_train = np.squeeze(correct_tensor_train.cpu().numpy())
        num_correct_train += np.sum(correct_train)

        loss = criterion(prediction.squeeze(), sentiment.float())
    
        # backward & optimize
        loss.backward()
        optimizer.step()

        if (idx % print_every) == 0:
            # Validation section
            model.eval()
            valid_losses = []
            valid_accuracy = []

            for idx_val, (val_sentence, val_sentiment, val_length) in enumerate(val_loader):
                num_correct_val = 0
                val_sentence, val_sentiment = val_sentence.to(device), val_sentiment.to(device)
                
                # forward
                val_prediction = model(val_sentence, val_length)

                # Pengukuran akurasi validasi
                # convert output probabilities to predicted class (0 or 1)
                pred_val = torch.round(val_prediction.squeeze())  # rounds to the nearest integer

                # compare predictions to true label
                correct_tensor_val = pred_val.eq(val_sentiment.float().view_as(pred_val))

                if device == 'cpu':
                    correct_val = np.squeeze(correct_tensor_val.numpy()) 
                else: 
                    correct_val = np.squeeze(correct_tensor_val.cpu().numpy())
                
                num_correct_val += np.sum(correct_val)
                
                val_loss = criterion(val_prediction.squeeze(), val_sentiment.float())
                valid_losses.append(val_loss.item())

            wandb.log({ "Train Loss": loss.item(), 
                        "Valid Loss": np.mean(valid_losses),
                        "Train Acc": num_correct_train/len(correct_train),
                        "Valid Acc": num_correct_val/len(correct_val)
                      })

            print("Epoch: {}/{}".format((epoch_id+1), epoch),
                "Step: {}/{}".format(idx, total_step),
                "Training Loss: {:.4f}".format(loss.item()),
                "Validation Loss: {:.4f}".format(np.mean(valid_losses)),
                "Training Accuracy:  {:.4f}".format(num_correct_train/len(correct_train)),
                "Validation Accuracy:  {:.4f}".format(num_correct_val/len(correct_val))
                )
