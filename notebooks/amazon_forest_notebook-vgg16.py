
# coding: utf-8

# # Planet: Understanding the Amazon deforestation from Space challenge

# Special thanks to the kernel contributors of this challenge (especially @anokas and @Kaggoo) who helped me find a starting point for this notebook.
# 
# The whole code including the `data_helper.py` and `keras_helper.py` files are available on github [here](https://github.com/EKami/planet-amazon-deforestation) and the notebook can be found on the same github [here](https://github.com/EKami/planet-amazon-deforestation/blob/master/notebooks/amazon_forest_notebook.ipynb)
# 
# **If you found this notebook useful some upvotes would be greatly appreciated! :) **

# Start by adding the helper files to the python path

# In[1]:

import sys

sys.path.append('../src')
sys.path.append('../tests')


# ## Import required modules

# In[2]:

import os
import gc
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import data_helper
from keras_helper import AmazonKerasClassifier
#from kaggle_data.downloader import KaggleDataDownloader

get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'retina'")


# Print tensorflow version for reuse (the Keras module is used directly from the tensorflow framework)

# In[3]:

tf.__version__


# ## Download the competition files
# Download the dataset files and extract them automatically with the help of [Kaggle data downloader](https://github.com/EKami/kaggle-data-downloader)

# In[4]:

competition_name = "planet-understanding-the-amazon-from-space"

train, train_u = "train-jpg.tar.7z", "train-jpg.tar"
test, test_u = "test-jpg.tar.7z", "test-jpg.tar"
test_additional, test_additional_u = "test-jpg-additional.tar.7z", "test-jpg-additional.tar"
test_labels = "train_v2.csv.zip"
destination_path = "C:/data/amazon/"
is_datasets_present = True

# If the folders already exists then the files may already be extracted
# This is a bit hacky but it's sufficient for our needs
datasets_path = data_helper.get_jpeg_data_files_paths(destination_path)
for dir_path in datasets_path:
    if os.path.exists(dir_path):
        is_datasets_present = True

if not is_datasets_present:
    # Put your Kaggle user name and password in a $KAGGLE_USER and $KAGGLE_PASSWD env vars respectively
    downloader = KaggleDataDownloader(os.getenv("KAGGLE_USER"), os.getenv("KAGGLE_PASSWD"), competition_name)
    
    train_output_path = downloader.download_dataset(train, destination_path)
    downloader.decompress(train_output_path, destination_path) # Outputs a tar file
    downloader.decompress(destination_path + train_u, destination_path) # Extract the content of the previous tar file
    os.remove(train_output_path) # Removes the 7z file
    os.remove(destination_path + train_u) # Removes the tar file
    
    test_output_path = downloader.download_dataset(test, destination_path)
    downloader.decompress(test_output_path, destination_path) # Outputs a tar file
    downloader.decompress(destination_path + test_u, destination_path) # Extract the content of the previous tar file
    os.remove(test_output_path) # Removes the 7z file
    os.remove(destination_path + test_u) # Removes the tar file
    
    test_add_output_path = downloader.download_dataset(test_additional, destination_path)
    downloader.decompress(test_add_output_path, destination_path) # Outputs a tar file
    downloader.decompress(destination_path + test_additional_u, destination_path) # Extract the content of the previous tar file
    os.remove(test_add_output_path) # Removes the 7z file
    os.remove(destination_path + test_additional_u) # Removes the tar file
    
    test_labels_output_path = downloader.download_dataset(test_labels, destination_path)
    downloader.decompress(test_labels_output_path, destination_path) # Outputs a csv file
    os.remove(test_labels_output_path) # Removes the zip file
else:
    print("All datasets are present.")


# ## Inspect image labels
# Visualize what the training set looks like

# In[5]:

import data_helper

train_jpeg_dir, test_jpeg_dir, test_jpeg_additional, train_csv_file = data_helper.get_jpeg_data_files_paths(destination_path)
labels_df = pd.read_csv(train_csv_file)
labels_df.head()


# Each image can be tagged with multiple tags, lets list all uniques tags

# In[6]:

# Print all unique tags
from itertools import chain
labels_list = list(chain.from_iterable([tags.split(" ") for tags in labels_df['tags'].values]))
labels_set = set(labels_list)
print("There is {} unique labels including {}".format(len(labels_set), labels_set))


# ### Repartition of each labels

# In[7]:

# Histogram of label instances
labels_s = pd.Series(labels_list).value_counts() # To sort them by count
fig, ax = plt.subplots(figsize=(16, 8))
sns.barplot(x=labels_s, y=labels_s.index, orient='h')


# ## Images
# Visualize some chip images to know what we are dealing with.
# Lets vizualise 1 chip for the 17 images to get a sense of their differences.

# In[8]:

images_title = [labels_df[labels_df['tags'].str.contains(label)].iloc[i]['image_name'] + '.jpg' 
                for i, label in enumerate(labels_set)]

plt.rc('axes', grid=False)
_, axs = plt.subplots(5, 4, sharex='col', sharey='row', figsize=(15, 20))
axs = axs.ravel()

for i, (image_name, label) in enumerate(zip(images_title, labels_set)):
    img = mpimg.imread(train_jpeg_dir + '/' + image_name)
    axs[i].imshow(img)
    axs[i].set_title('{} - {}'.format(image_name, label))


# # Image Resize
# Define the dimensions of the image data trained by the network. Due to memory constraints we can't load in the full size 256x256 jpg images. Recommended resized images could be 32x32, 64x64, or 128x128.

# In[9]:

img_resize = (64, 64) # The resize size of each image


# # Data preprocessing
# Preprocess the data in order to fit it into the Keras model.
# 
# Due to the hudge amount of memory the resulting matrices will take, the preprocessing will be splitted into several steps:
#     - Preprocess training data (images and labels) and train the neural net with it
#     - Delete the training data and call the gc to free up memory
#     - Preprocess the first testing set
#     - Predict the first testing set labels
#     - Delete the first testing set
#     - Preprocess the second testing set
#     - Predict the second testing set labels and append them to the first testing set
#     - Delete the second testing set

# In[10]:

x_train, y_train, y_map = data_helper.preprocess_train_data(train_jpeg_dir, train_csv_file, img_resize)
# Free up all available memory space after this heavy operation
gc.collect();


# In[11]:

print("x_train shape: {}".format(x_train.shape))
print("y_train shape: {}".format(y_train.shape))
y_map


# ## Create a checkpoint
# 
# Creating a checkpoint saves the best model weights across all epochs in the training process. This ensures that we will always use only the best weights when making our predictions on the test set rather than using the default which takes the final score from the last epoch. 

# In[12]:

from tensorflow.contrib.keras.api.keras.callbacks import ModelCheckpoint

filepath="weights.best.vgg16.top.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)


# ## Choose Hyperparameters
# 
# Choose your hyperparameters below for training. 

# In[13]:

validation_split_size = 0.2
batch_size = 128
bottleneck_features_path = 'bottleneck_features_64_vgg.npy'


# ## Define and Train model
# 
# Here we define the model and begin training. 
# 
# Note that we have created a learning rate annealing schedule with a series of learning rates as defined in the array `learn_rates` and corresponding number of epochs for each `epochs_arr`. Feel free to change these values if you like or just use the defaults. 

# In[14]:

classifier = AmazonKerasClassifier()

classifier.save_vgg16_bottleneck_features(x_train, bottleneck_features_path, batch_size=batch_size)
gc.collect()


# In[15]:

train_data = np.load(open(bottleneck_features_path, 'rb'))


# In[16]:

train_data.shape
print('train_data.shape[1:]: {}'.format(train_data.shape[1:]))


# In[17]:

x_train.shape


# In[18]:

classifier = AmazonKerasClassifier()
#classifier.add_conv_layer(img_resize)
classifier.add_flatten_layer(input_shape=train_data.shape[1:])
classifier.add_ann_layer(len(y_map))

train_losses, val_losses = [], []
#epochs_arr = [10, 5, 5]
'''epochs_arr = [2, 1, 1]
learn_rates = [0.001, 0.0001, 0.00001]
for learn_rate, epochs in zip(learn_rates, epochs_arr):
    tmp_train_losses, tmp_val_losses, fbeta_score = classifier.train_model(train_data, y_train, learn_rate, epochs, 
                                                                           batch_size, validation_split_size=validation_split_size, 
                                                                           train_callbacks=[checkpoint])
    train_losses += tmp_train_losses
    val_losses += tmp_val_losses
'''


# ## Monitor the results

# Check that we do not overfit by plotting the losses of the train and validation sets

# In[21]:

plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.legend();


# Look at our fbeta_score

# In[22]:

fbeta_score


# ## Define Full Model

# In[ ]:




# ### Load Best Weights

# Here you should load back in the best weights that were automatically saved by ModelCheckpoint during training

# In[19]:

classifier.load_weights(filepath)
print("Weights loaded")


# In[24]:

import sys

sys.path.append('../src')
from keras_helper import AmazonKerasClassifier


train_losses, val_losses = [], []
#epochs_arr = [10, 5, 5]
epochs_arr = [2, 1, 1]
learn_rates = [0.001, 0.0001, 0.00001]
for learn_rate, epochs in zip(learn_rates, epochs_arr):
    tmp_train_losses, tmp_val_losses, fbeta_score = classifier.train_vgg16_full_model(train_data, y_train, img_resize, 3, learn_rate, epochs, 
                                                                           batch_size, validation_split_size=validation_split_size, 
                                                                           train_callbacks=[checkpoint])
    train_losses += tmp_train_losses
    val_losses += tmp_val_losses


# In[ ]:




# Before launching our predictions lets preprocess the test data and delete the old training data matrices

# In[23]:

del x_train, y_train
gc.collect()

x_test, x_test_filename = data_helper.preprocess_test_data(test_jpeg_dir, img_resize)
# Predict the labels of our x_test images
predictions = classifier.predict(x_test)


# Now lets launch the predictions on the additionnal dataset (updated on 05/05/2017 on Kaggle)

# In[21]:

del x_test
gc.collect()

x_test, x_test_filename_additional = data_helper.preprocess_test_data(test_jpeg_additional, img_resize)
new_predictions = classifier.predict(x_test)

del x_test
gc.collect()
predictions = np.vstack((predictions, new_predictions))
x_test_filename = np.hstack((x_test_filename, x_test_filename_additional))
print("Predictions shape: {}\nFiles name shape: {}\n1st predictions entry:\n{}".format(predictions.shape, 
                                                                              x_test_filename.shape,
                                                                              predictions[0]))


# Before mapping our predictions to their appropriate labels we need to figure out what threshold to take for each class.
# 
# To do so we will take the median value of each classes.

# In[22]:

# For now we'll just put all thresholds to 0.2 
thresholds = [0.2] * len(labels_set)

# TODO complete
tags_pred = np.array(predictions).T
_, axs = plt.subplots(5, 4, figsize=(15, 20))
axs = axs.ravel()

for i, tag_vals in enumerate(tags_pred):
    sns.boxplot(tag_vals, orient='v', palette='Set2', ax=axs[i]).set_title(y_map[i])


# Now lets map our predictions to their tags and use the thresholds we just retrieved

# In[23]:

predicted_labels = classifier.map_predictions(predictions, y_map, thresholds)


# Finally lets assemble and visualize our prediction for the test dataset

# In[24]:

tags_list = [None] * len(predicted_labels)
for i, tags in enumerate(predicted_labels):
    tags_list[i] = ' '.join(map(str, tags))

final_data = [[filename.split(".")[0], tags] for filename, tags in zip(x_test_filename, tags_list)]


# In[25]:

final_df = pd.DataFrame(final_data, columns=['image_name', 'tags'])
final_df.head()


# In[26]:

tags_s = pd.Series(list(chain.from_iterable(predicted_labels))).value_counts()
fig, ax = plt.subplots(figsize=(16, 8))
sns.barplot(x=tags_s, y=tags_s.index, orient='h');


# If there is a lot of `primary` and `clear` tags, this final dataset may be legit...

# And save it to a submission file

# In[27]:

final_df.to_csv('../submission_file.csv', index=False)
classifier.close()


# That's it, we're done!
