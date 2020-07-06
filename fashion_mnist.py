#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.datasets import fashion_mnist
import numpy as np


# # LOADING OUR DATASET

# In[2]:


dataset = fashion_mnist.load_data()


# In[3]:


len(dataset)


# In[4]:


train , test = dataset


# In[5]:


len(train)


# In[6]:


X_train , y_train = train


# In[7]:


X_train.shape


# In[8]:


X_test , y_test = test


# In[9]:


X_test.shape


# In[10]:


img1 = X_train[7]


# In[11]:


img1.shape


# In[12]:


import cv2


# In[13]:


img1_label = y_train[7]


# In[14]:


img1_label


# In[15]:


img1.shape


# In[16]:


import matplotlib.pyplot as plt


# In[17]:


plt.imshow(img1 , cmap='gray')


# In[18]:


img1.shape


# In[19]:


img1_1d = img1.reshape(28*28)


# In[20]:


img1_1d.shape


# In[21]:


X_train.shape


# In[22]:


X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)


# In[23]:


X_train.shape


# In[24]:


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


# In[25]:


X_train.shape


# In[26]:


y_train.shape


# In[27]:


# Normalizing the Dataset to help with the training
# We rescale these image by dividing each pixel’s value by 255. 
X_train = X_train/255
X_test = X_test/255


# # USING ONE HOT ENCODING TO NORMALIZE THE DATA

# In[28]:


from keras.utils.np_utils import to_categorical


# In[29]:


y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)


# In[30]:


y_train_cat


# In[31]:


y_test_cat


# In[32]:


y_train_cat[7]


# # BUILDING THE CONVOLUTION NETWORK

# In[33]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten


# In[34]:


model = Sequential()


# In[35]:


model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))


# In[36]:


model.add(MaxPooling2D((2, 2)))
model.add(Flatten())


# In[37]:


model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))


# In[38]:


model.add(Dense(units=10, activation='softmax'))


# # MODEL SUMMARY

# In[39]:


model.summary()


# # COMPILING THE MODEL

# In[40]:


from keras.optimizers import RMSprop


# In[41]:


model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', 
             metrics=['accuracy']
             )


# # FIT THE MODEL

# In[42]:



history = model.fit(X_train, y_train_cat, epochs=10, batch_size=32, validation_data=(X_test, y_test_cat))


# In[44]:


# make predictions on the test set
model.predict(X_test)


# In[45]:


y_test[0]


# In[46]:


test_img = X_test[0].reshape(28*28)


# # EVALUATING THE MODEL

# In[56]:


score = model.evaluate(X_test, y_test_cat, verbose=0)
print("Loss: {:.4f}" .format(score[0] * 100))
print("Accuracy: {:.4f}" .format(score[1] * 100))


# In[47]:


test_img.shape


# In[ ]:


X_test.shape


# In[ ]:


X_test[0].shape


# In[ ]:




