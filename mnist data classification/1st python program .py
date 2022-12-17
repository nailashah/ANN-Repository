#!/usr/bin/env python
# coding: utf-8

# In[3]:


from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# In[4]:


train_images.shape


# In[5]:


len(train_labels)


# In[6]:


train_labels


# In[7]:


len(test_labels)


# In[8]:


test_labels


# In[9]:


from keras import models
from keras import layers
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))


# In[10]:


network.compile(optimizer='rmsprop',
loss='categorical_crossentropy',
metrics=['accuracy'])


# In[11]:


train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255


# In[12]:


from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


# In[13]:


network.fit(train_images, train_labels, epochs=5, batch_size=128)


# In[14]:


test_loss, test_acc = network.evaluate(test_images, test_labels)


# In[15]:


print('test_acc:', test_acc)


# In[16]:


network.evaluate(test_images, test_labels)


# In[1]:


60000/128


# In[ ]:




