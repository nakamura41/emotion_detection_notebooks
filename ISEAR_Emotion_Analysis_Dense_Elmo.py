#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3"
os.environ["TFHUB_CACHE_DIR"]="tfhub_modules"


import tensorflow as tf
from keras import backend as K
print(tf.__version__)
print(K.tensorflow_backend._get_available_gpus())


# In[11]:


import os
import logging
import pandas as pd
import numpy as np
import tensorflow_hub as hub
from keras.utils import to_categorical
from keras.engine import Layer
import keras.layers as layers
from keras.models import Model, load_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')


# Initialize session
sess = tf.Session()
K.set_session(sess)


# In[3]:


from numpy.random import seed
from tensorflow import set_random_seed

RANDOM_SEED = 20190101

def set_random(random_seed):
    seed(random_seed)
    set_random_seed(random_seed)

set_random(RANDOM_SEED)


# In[4]:


class ISEARDataset(object):
  FILENAME = "data/isear_databank.csv"
  EMOTION_CLASSES = ["anger", "disgust", "fear", "guilt", "joy", "sadness", "shame"]
  EMOTION_CLASSES_DICT = {"anger": 0, "disgust": 1, "fear": 2, "guilt": 3, "joy": 4, "sadness": 5, "shame": 6}
  RANDOM_STATE = 41

  def get_classes(self):
    return self.EMOTION_CLASSES

  def get_classes_dict(self):
    return self.EMOTION_CLASSES_DICT

  def __load_data_file(self):
    data = pd.read_csv(self.FILENAME)
    data["emotion"] = data["Field1"]
    data["text"] = data["SIT"]
    return data[["text", "emotion"]]

  def load_data(self):
    train_data = None
    test_data = None
    data = self.__load_data_file()
    train_data, test_data = train_test_split(data, test_size=0.3, random_state=self.RANDOM_STATE, stratify=data["emotion"].values)
    return train_data, test_data


# In[5]:


isear_dataset = ISEARDataset()
train_data, test_data = isear_dataset.load_data()
train_data, valid_data = train_test_split(train_data, test_size=0.1, random_state=200, stratify=train_data.emotion)

logging.debug("train_data.shape: (%d, %d)" % train_data.shape)
logging.debug("valid_data.shape: (%d, %d)" % valid_data.shape)
logging.debug("test_data.shape: (%d, %d)" % test_data.shape)


# In[6]:


dic = isear_dataset.get_classes_dict()
labels = isear_dataset.get_classes()
n_classes = len(labels)
logging.debug("class dictionary: %s" % dic)
logging.debug("class labels: %s" % labels)
logging.debug("number of bins: %s" % n_classes)

for emotion in labels:
  train_data.loc[train_data.emotion == emotion, "emotion_int"] = dic[emotion]
  valid_data.loc[valid_data.emotion == emotion, "emotion_int"] = dic[emotion]
  test_data.loc[test_data.emotion == emotion, "emotion_int"] = dic[emotion]

bins = list(range(0, n_classes + 1))
logging.debug("bins: %s" % bins)
hist, _ = np.histogram(train_data["emotion_int"], bins=bins)

y_pos = np.arange(len(labels))

plt.bar(y_pos, hist, align='center', alpha=0.5)
plt.xticks(y_pos, labels)
plt.ylabel('Number')
plt.title('Emotions')

plt.show()


# In[7]:


X_train = np.array(train_data.text, dtype=object)[:, np.newaxis]
X_val = np.array(valid_data.text, dtype=object)[:, np.newaxis]
X_test = np.array(test_data.text, dtype=object)[:, np.newaxis]

y_train = to_categorical(np.asarray(train_data.emotion.apply(lambda x:dic[x])))
y_val = to_categorical(np.asarray(valid_data.emotion.apply(lambda x:dic[x])))
y_test = to_categorical(np.asarray(test_data.emotion.apply(lambda x:dic[x])))

logging.debug('Shape of X train, validation and test tensor: %s, %s, %s' % (X_train.shape, X_val.shape, X_test.shape))
logging.debug('Shape of label train, validation and test tensor: %s, %s, %s' % (y_train.shape, y_val.shape, y_test.shape))


# In[8]:


# Create a custom layer that allows us to update weights (lambda layers do not have trainable parameters!)

class ElmoEmbeddingLayer(Layer):
    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable=True
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable,
                               name="{}_module".format(self.name))

        self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1),
                      as_dict=True,
                      signature='default',
                      )['default']
        return result

    def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs, '--PAD--')

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dimensions)


# In[9]:


# Function to build model
def build_model():
  input_text = layers.Input(shape=(1,), dtype="string")
  embedding = ElmoEmbeddingLayer()(input_text)
  dense = layers.Dense(256, activation='relu')(embedding)
  pred = layers.Dense(n_classes, activation='sigmoid')(dense)

  model = Model(inputs=[input_text], outputs=pred)

  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  model.summary()

  return model


# In[ ]:


import datetime

start_time = datetime.datetime.now()

# Build and fit
model = build_model()

model.fit(X_train,
          y_train,
          validation_data=(X_val, y_val),
          epochs=1,
          batch_size=32)

end_time = datetime.datetime.now()

def days_hours_minutes_seconds(td):
  return td.days, td.seconds//3600, (td.seconds//60)%60, (td.seconds%60)

elapsed_time = end_time - start_time
logging.debug("%d days, %d hours, %d minutes, %d seconds elapsed" % (days_hours_minutes_seconds(elapsed_time)))


# In[ ]:


model.save('isear_dense_elmo_model.h5')
model.save_weights('isear_dense_elmo_weights.h5')

logging.debug("model and weights is successfully saved")

