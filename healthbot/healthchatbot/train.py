import json
import numpy as np
from utils import tokenize, stemmingg, words_container
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet

with open("datafile.json", 'r') as f:
    data_words = json.load(f)

all_words = []
tags = []
xy = []

for obj in data_words['intents']:
    tag = obj['tag']
    tags.append(tag)
    for pattern in obj['patterns']:
        word = tokenize(pattern)
        all_words.extend(word)
        xy.append((word, tag))

ignored_list = ['?', '!', '.', ',']

all_words = [stemmingg(w) for w in all_words if w not in ignored_list]

all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
Y_train = []

for (sentence, tag) in xy:
    words = words_container(sentence, all_words)
    X_train.append(words)

    labels = tags.index(tag)
    Y_train.append(labels)


X_train = np.array(X_train)
Y_train = np.array(Y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train

    #dataset index
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples
  
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(X_train[0])
learning_rate = 0.001
num_epochs = 1000

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        outputs = model(words)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'epoch [{epoch+1}/{num_epochs}], loss : {loss.item():.4f}')

print(f'final loss, loss : {loss.item():.4f}')

data = {
    'model_state' : model.state_dict(),
    'input_size' : input_size,
    'output_size' : output_size,
    'hidden_size' : hidden_size,
    'all_words' : all_words,
    'tags' : tags
}

FILE = "trained_weights.pth"
torch.save(data, FILE)

print(f"training complete, data saved in the {FILE} file")