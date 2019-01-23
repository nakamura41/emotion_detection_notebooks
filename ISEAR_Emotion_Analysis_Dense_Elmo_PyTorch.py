#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["TFHUB_CACHE_DIR"]="tfhub_modules"

torch.manual_seed(1)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


# In[2]:


from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.modules.elmo import batch_to_ids

NUM_WORDS = 20000

class ISEARDataset(object):
    FILENAME = "data/isear_databank.csv"
    EMOTION_CLASSES = ["anger", "disgust", "fear", "guilt", "joy", "sadness", "shame"]
    EMOTION_CLASSES_DICT = {"anger": 0, "disgust": 1, "fear": 2, "guilt": 3, "joy": 4, "sadness": 5, "shame": 6}
    RANDOM_STATE = 41
  
    def get_classes(self):
        return self.EMOTION_CLASSES
  
    def get_classes_dict(self):
        return self.EMOTION_CLASSES_DICT
    
    def _tokens_to_texts(self, tokens):
        texts = []
        for token in tokens:
            texts.append([t.text for t in token])
        return texts

    def _sequence_texts(self, texts):
        tokenizer = WordTokenizer()
        tokens = tokenizer.batch_tokenize(texts)
        sequences = batch_to_ids(self._tokens_to_texts(tokens))
        return sequences

    def __init__(self, n_items=0):
        data = pd.read_csv(self.FILENAME)

        if n_items > 0:
            data = data.iloc[0:n_items,:]

        data["text"] = data["SIT"]
        data["emotion"] = data["Field1"]

        for emotion in self.get_classes():
            data.loc[data["emotion"] == emotion, "emotion_int"] = self.get_classes_dict()[emotion]

        self.X = self._sequence_texts(data["text"]).numpy()
        self.y = data["emotion_int"].values


# In[3]:


dataset = ISEARDataset()
# dataset = ISEARDataset(200)
X_train, X_test, y_train, y_test = train_test_split(dataset.X, dataset.y, test_size=0.3, random_state=dataset.RANDOM_STATE, stratify=dataset.y)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=dataset.RANDOM_STATE, stratify=y_train)


# In[4]:


print("X_train.shape: (%d, %d, %d)" % X_train.shape)
print("y_train.shape: (%d)" % y_train.shape)

print("X_valid.shape: (%d, %d, %d)" % X_valid.shape)
print("y_valid.shape: (%d)" % y_valid.shape)

print("X_test.shape: (%d, %d, %d)" % X_test.shape)
print("y_test.shape: (%d)" % y_test.shape)


# In[5]:


np.save("pytorch_isear_X_train_elmo.npy", X_train)
np.save("pytorch_isear_X_valid_elmo.npy", X_valid)
np.save("pytorch_isear_X_test_elmo.npy", X_test)

np.save("pytorch_isear_y_train_elmo.npy", y_train)
np.save("pytorch_isear_y_valid_elmo.npy", y_valid)
np.save("pytorch_isear_y_test_elmo.npy", y_test)


# In[6]:


X_train = np.load("pytorch_isear_X_train_elmo.npy")
X_valid = np.load("pytorch_isear_X_valid_elmo.npy")
X_test = np.load("pytorch_isear_X_test_elmo.npy")

y_train = np.load("pytorch_isear_y_train_elmo.npy")
y_valid = np.load("pytorch_isear_y_valid_elmo.npy")
y_test = np.load("pytorch_isear_y_test_elmo.npy")


# In[7]:


dic = dataset.get_classes_dict()
labels = dataset.get_classes()
n_classes = len(labels)
print("class dictionary: %s" % dic)
print("class labels: %s" % labels)
print("number of bins: %s" % n_classes)


# In[8]:


bins = list(range(0, n_classes + 1))
print("bins:", bins)


# In[9]:


from torch import autograd

def make_one_hot(labels, C=2):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.
    
    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size. 
        Each value is an integer representing correct classification.
    C : integer. 
        number of classes in labels.
    
    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x H x W, where C is class number. One-hot encoded.
    '''
    one_hot = torch.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_()
    target = one_hot.scatter_(1, labels.data, 1)
    
    target = autograd.Variable(target)
        
    return target
  
y = torch.LongTensor(y_train).view(-1, 1, 1, 1)
print("y.shape:", y.shape)
y_onehot = make_one_hot(y, C=7)
print("y_onehot.shape:", y_onehot.shape)


# In[10]:


import torch
from torch.utils import data

class ISEAR_Tensor_Dataset(data.TensorDataset):
  
  def __init__(self, text, emotion, num_class=2):
    X = torch.LongTensor(text)
    y = torch.LongTensor(emotion).view(-1, 1, 1, 1)
    y_onehot = make_one_hot(y, num_class)
    y_onehot = y_onehot.view(y_onehot.shape[0], y_onehot.shape[1])
    tensors = []
    tensors.append(X)
    tensors.append(y_onehot)
    super().__init__(*tensors)


# In[11]:


train_dataset = ISEAR_Tensor_Dataset(X_train, y_train, num_class=7)
valid_dataset = ISEAR_Tensor_Dataset(X_valid, y_valid, num_class=7)
test_dataset = ISEAR_Tensor_Dataset(X_test, y_test, num_class=7)


# In[12]:


print("train_dataset.tensors[0].shape:", train_dataset.tensors[0].shape)
print("train_dataset.tensors[1].shape:", train_dataset.tensors[1].shape)


# In[13]:


print("train_dataset length:", len(train_dataset))
print("valid_dataset length:", len(valid_dataset))
print("test_dataset length:", len(test_dataset))


# In[14]:


batch_size = 50

train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


# ### Load Elmo

# In[15]:


from allennlp.modules.elmo import Elmo, batch_to_ids

options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

# Compute two different representation for each token.
# Each representation is a linear weighted combination for the
# 3 layers in ELMo (i.e., charcnn, the outputs of the two BiLSTM))
elmo = Elmo(options_file, weight_file, num_output_representations=2, dropout=0)
elmo = elmo.to(device)


# In[16]:


shape = train_dataset[0][0].shape
print("shape:", shape)
experiment_X_train = train_dataset[0][0].view(1, shape[0], shape[1]).to(device)
print("experiment_X_train.shape:", experiment_X_train.shape)

embedded = elmo(experiment_X_train)
print("embedded shape:", embedded['elmo_representations'][0].shape)


# In[17]:


import scipy

sentences = [
  ["I", "ate", "an", "apple", "for", "breakfast"], 
  ["I", "ate", "a", "carrot", "for", "breakfast"]
]
character_ids = batch_to_ids(sentences)

embeddings = elmo(character_ids.to(device))
embeddings['elmo_representations'][0][0]


# In[18]:


vector1 = embeddings['elmo_representations'][0][0].cpu().detach().numpy()
vector2 = embeddings['elmo_representations'][0][1].cpu().detach().numpy()
scipy.spatial.distance.cosine(vector1[3], vector2[3]) # cosine distance between "apple" and "carrot" in the last layer


# ## Neural Network Architecture 

# In[ ]:


import torch.nn as nn
import torch.nn.functional as F

class ElmoClassifier(nn.Module):

    def __init__(self, device, elmo_embedding, input_dim, embedding_dim=1024, linear_dim=256,
                 output_dim=7):

        super(ElmoClassifier, self).__init__()
        
        self.device = device
        self.embedding = elmo_embedding
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.linear_dim = linear_dim
        self.output_dim = output_dim
        
        self.linear = nn.Linear(in_features=self.embedding_dim, out_features=self.linear_dim)
        self.linear_pred = nn.Linear(in_features=self.linear_dim, out_features=self.output_dim)

    def __shape_to_string(self, shape):
        return str(shape)
    
    def summary(self, input_size):
        sentence = torch.LongTensor(np.zeros(input_size)).to(self.device)
        print("{:<20}  {:<25}".format("input", self.__shape_to_string(sentence.shape)))
        embeds = self.embedding(sentence)['elmo_representations'][0]
        print("{:<20}  {:<25}".format("embedding", self.__shape_to_string(embeds.shape)))
        avg_embeds = embeds.mean(dim=1)
        print("{:<20}  {:<25}".format("avg_embeds", self.__shape_to_string(avg_embeds.shape)))
        dense = self.linear(avg_embeds)
        print("{:<20}  {:<25}".format("dense", self.__shape_to_string(dense.shape)))
        output = F.softmax(self.linear_pred(dense))
        print("{:<20}  {:<25}".format("output", self.__shape_to_string(output.shape)))

    def forward(self, sentence):
        embeds = self.embedding(sentence)['elmo_representations'][0]
        avg_embeds = embeds.mean(dim=1)
        dense = self.linear(avg_embeds)
        output = F.softmax(self.linear_pred(dense))
        return output


# In[ ]:


input_dim = X_train.shape[1]
embedding_dim = 1024
linear_dim = 256
output_dim = 7

model = ElmoClassifier(device, elmo, input_dim, embedding_dim, linear_dim, output_dim)
model = model.to(device)

model.summary([1, 201, 50])


# In[ ]:


train_loader.__dict__


# In[ ]:


def train(epoch, model, data_loader, device, optimiser, loss_fn=nn.BCELoss(), log_interval=100):
    #####################
    # Train model
    #####################

    # switch model to training mode, clear gradient accumulators
    model.train()
    # model.hidden = model.init_hidden()
    
    train_loss = 0
    total_correct = 0

    all_pred = []
    all_target = []

    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimiser.zero_grad()
        output = model(data)
        
        loss = loss_fn(output, target)
        loss.backward()
        optimiser.step()
        
        train_loss += loss.data # sum up batch loss
        
        pred = output.max(1, keepdim=True)[1]
        pred = pred.view(pred.size(0))

        target = target.max(1, keepdim=True)[1]
        target = target.view(target.size(0))

        correct = pred.eq(target.view_as(pred)).sum()
        total_correct += correct

        all_pred += pred.cpu().numpy().tolist()
        all_target += target.cpu().numpy().tolist()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}\t/\t{}\t({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data))
    
    print('Train: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, total_correct, len(data_loader.dataset),
        100. * total_correct / len(data_loader.dataset)))
    
    return train_loss, all_pred, all_target


# In[ ]:


def evaluate(model, data_loader, device, loss_fn=nn.BCELoss()):
    #####################
    # Evaluation model
    #####################
    model.eval()
    eval_loss = 0
    total_correct = 0

    all_pred = []
    all_target = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            eval_loss += loss_fn(output, target).data # sum up batch loss
            
            pred = output.max(1, keepdim=True)[1]
            pred = pred.view(pred.size(0))
            
            target = target.max(1, keepdim=True)[1]
            target = target.view(target.size(0))
            
            correct = pred.eq(target.view_as(pred)).sum()
            total_correct += correct

            all_pred += pred.cpu().numpy().tolist()
            all_target += target.cpu().numpy().tolist()

    print('Evaluate: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        eval_loss, total_correct, len(data_loader.dataset),
        100. * total_correct / len(data_loader.dataset)))
    
    return eval_loss, all_pred, all_target


# In[ ]:


loss_fn = nn.BCELoss()
log_interval = 10
max_epochs = 30
learning_rate = 1e-3
optimiser = optim.Adam(model.parameters(), lr=learning_rate)

train_loss_hist = np.zeros(max_epochs)
eval_loss_hist = np.zeros(max_epochs)

for epoch in range(max_epochs):
  
    train_loss, train_pred, train_target = train(epoch, model, train_loader, device, optimiser, loss_fn, log_interval)
    train_loss_hist[epoch] = train_loss
    
    valid_loss, valid_pred, valid_target = evaluate(model, valid_loader, device, loss_fn=nn.BCELoss())
    eval_loss_hist[epoch] = valid_loss


# In[ ]:


model_path = 'pytorch_isear_dense_elmo_model.h5'
weight_path = 'pytorch_isear_dense_elmo_weights.h5'

torch.save(model.state_dict(), weight_path)
torch.save(model, model_path)


# In[ ]:


test_loss, test_pred, test_target = evaluate(model, test_loader, device, loss_fn)


# In[ ]:


from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

cf_matrix = confusion_matrix(test_target, test_pred)

df_cm = pd.DataFrame(
    cf_matrix, index=labels, columns=labels, 
)

print(df_cm)


# In[ ]:


print(print_confusion_matrix(cf_matrix, class_names=labels))


# In[ ]:


test_accuracy = accuracy_score(test_target, test_pred)
print("test accuracy:", test_accuracy)


# ### Performance score for each classes

# In[ ]:


precision, recall, fscore, support = precision_recall_fscore_support(test_target, test_pred)
score_dict = {
  "precision": precision.round(4),
  "recall": recall.round(4),
  "f1-score": fscore.round(4),
  "support": support.round(4)
}
score_df = pd.DataFrame(score_dict, index=labels)
score_df


# ### Cohen Kappa Score

# In[ ]:


from sklearn.metrics import cohen_kappa_score

kappa_score = cohen_kappa_score(test_target, test_pred)
print("kappa:", kappa_score)


# In[ ]:






