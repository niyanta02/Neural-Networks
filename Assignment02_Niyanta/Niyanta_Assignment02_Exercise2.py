#!/usr/bin/env python
# coding: utf-8

# # Write a Python program to train CIFAR10 dataset using Backpropagation. Use TensorFlow.

# In[1]:


import tensorflow as tf
from tensorflow import keras
tf.__version__


# In[2]:


keras.__version__


# In[9]:


cifar10 = keras.datasets.cifar10
(X_train_full, y_train_full), (X_test, y_test) = cifar10.load_data()


# In[11]:


y_train


# In[12]:


X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.0


# In[13]:


print(X_train_full.shape, y_train_full.shape, X_test.shape, y_test.shape)


# In[14]:


class_names = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "hourse", "ship", "truck"]


# In[16]:


print(y_train[0])
class_names[int(y_train[0])]


# In[18]:


model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[32,32,3]))
model.add(keras.layers.Dense(400, activation="relu"))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(30, activation="softmax"))


# In[19]:


model.summary()


# In[20]:


hidden1 = model.layers[1]
hidden1.name
model.get_layer('dense_3') is hidden1


# In[21]:


weights, biases = hidden1.get_weights()
weights


# In[24]:


print(weights.shape)
print(biases)

print(biases.shape)


# In[25]:


model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])


# In[26]:


history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))


# In[27]:


import pandas as pd
import matplotlib.pyplot as plt

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 3) # set the vertical range to [0-1]
plt.show()


# In[28]:


model3 = keras.models.Sequential()
model3.add(keras.layers.Flatten(input_shape=[32, 32, 3]))
model3.add(keras.layers.Dense(400, activation="relu"))
model3.add(keras.layers.Dense(300, activation="relu"))
model3.add(keras.layers.Dense(100, activation="relu"))
model3.add(keras.layers.Dense(10, activation="softmax"))


# In[29]:


model3.compile(loss="sparse_categorical_crossentropy",
              optimizer="Adam",
              metrics=["accuracy"])


# In[30]:


history3 = model3.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))


# In[31]:


model4 = keras.models.Sequential()
model4.add(keras.layers.Flatten(input_shape=[32, 32, 3]))
model4.add(keras.layers.Dense(400, activation="relu"))
model4.add(keras.layers.Dense(300, activation="relu"))
model4.add(keras.layers.Dense(100, activation="relu"))                     
model4.add(keras.layers.Dense(10, activation="softmax"))


# In[32]:


model4.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(lr=1e-3),
              metrics=["accuracy"])

history4 = model4.fit(X_train, y_train, epochs=20, batch_size= 10,  validation_data=(X_valid, y_valid))


# In[33]:


import pandas as pd
import matplotlib.pyplot as plt

pd.DataFrame(history4.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 2) # set the vertical range to [0-1]
plt.show()


# In[34]:


model4.evaluate(X_test, y_test)


# In[35]:


X_new = X_test[:3]
y_proba = model4.predict(X_new)
y_proba.round(2)


# In[36]:


y_pred = model4.predict(X_new)
y_pred


# In[37]:


import numpy as np
classes_x=np.argmax(y_pred, axis=1)


# In[38]:


y_new = y_test[:3]
y_new


# In[39]:


#Tuning


# In[40]:


def model_builder(hp):
    
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(32, 32, 3)))

      # Tune the number of units in the first Dense layer
      # Choose an optimal value between 32-512
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    model.add(keras.layers.Dense(units=hp_units, activation='relu'))
    model.add(keras.layers.Dense(10))

      # Tune the learning rate for the optimizer
      # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

    return model


# In[42]:


import keras_tuner as kt
tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3
                     )


# In[46]:


stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)


# In[47]:


tuner.search(X_train_full, y_train_full, epochs=20, validation_split=0.2, callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")


# In[48]:


# Build the model with the optimal hyperparameters and train it on the data for 50 epochs
model = tuner.hypermodel.build(best_hps)
history = model.fit(X_train, y_train, epochs=20, batch_size= 10,  validation_data=(X_valid, y_valid))

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))


# In[49]:


hypermodel = tuner.hypermodel.build(best_hps)

# Retrain the model
hypermodel.fit(X_train, y_train, epochs=best_epoch, batch_size= 10,  validation_data=(X_valid, y_valid))


# In[50]:


eval_result = hypermodel.evaluate(X_test, y_test)
print("[test loss, test accuracy]:", eval_result)

