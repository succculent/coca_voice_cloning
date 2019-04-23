import numpy as np
import glob
import tensorflow as tf
from scipy.io.wavfile import read
import keras
import time
import tensorflow_hub as hub
import scipy as sp
from keras.models import load_model 

'''
  This function loads txt numpy array and return a 3D Numpy array containing them
'''
def load_txt_np():
  data_path = '/content/gdrive/Team Drives/Neural Network/TIMIT/TRAIN/*/*/*_txt.npy'
  data_path_test = '/content/gdrive/Team Drives/Neural Network/TIMIT/TEST/*/*/*_txt.npy'

  data = np.array([[]])
  filenames = glob.glob(data_path)
  filenames += glob.glob(data_path_test)
  count = 0
  print("Loading " + str(len(filenames)) + " text files")
  total_t = 0
  t = time.time()
  for filename in filenames:
    count += 1
    #print(count)
    np_array = np.load(filename)
    np_array = np.array([[np_array]])
    #print(np_array)
    #data.append(np_array)
    if count == 1:
      data = np_array
    else:
      data = np.vstack((data, np_array))
    if(count % (len(filenames)/10) == 0):
      print("Loaded  " + str(count) + " files and it took " + str(time.time() - t) + " second for this iteration")
      print(data.shape)
      total_t += time.time() - t
      t = time.time()
  np.save('/content/gdrive/Team Drives/Neural Network/TIMIT/text', data)
  print("Done, took total of " + str(total_t) + " seconds")
  return data
    

def normalizeAudio(labels):
    max_length = 0
    o_length = 0

    for i in range(0, len(labels)):
      y = len(labels[i])
      #list of numpy arrays of variable size (audio)
      #x, y = labels[i].shape
      if (y>max_length):
        max_length = y #double check if it is y or x lMAO

    new_labels = list()

    for i in range(0, len(labels)):
      x = len(labels[i])
      temp = np.zeros(max_length)
      #x, y = labels[i].shape
      for j in range(0, x):
  #			for k in range(0, y):
        temp[j] = labels[i][j]
      new_labels.append(temp)
    print(data.shape)
    return new_labels

  
def hi():
  now = time.time()
  filename = '/content/gdrive/Team Drives/Neural Network/TIMIT/wav1.npy'
  first = np.load(filename)
  print(first.shape)
  filename = '/content/gdrive/Team Drives/Neural Network/TIMIT/wav2.npy'
  second = np.load(filename)
  print(second.shape)
  print("Finished loading files")
  
  #This is the part to connect them together.--------
  first = np.ndarray.tolist(first)
  print("converted first to list")
  second = np.ndarray.tolist(second)
  print("convereted second to list")
  first.extend(second)
  
  #first = np.concatenate((first, second))
  #first = np.vstack((first, second))
  
  #--------------------------------------------------
  
  np.save('/content/gdrive/Team Drives/Neural Network/TIMIT/wav.npy', first)
  print("Done, took " + str(time.time() - now) + " seconds to load")
  return first


'''
  This function loads wav numpy array files, and save and returns numpy array containing them.
'''
def load_wav_np():
  data_path = '/content/gdrive/Team Drives/Neural Network/TIMIT/TRAIN/*/*/*_wav.npy'
  data_path_test = '/content/gdrive/Team Drives/Neural Network/TIMIT/TEST/*/*/*_wav.npy'
  
  #
  MAXLEN = 124621
 
  data = np.array([])
  filenames = glob.glob(data_path)
  filenames += glob.glob(data_path_test)
  
  count = 0
  print("Loading " + str(len(filenames)) + " wav files")
  total_t = 0
  t = time.time()
  for filename in filenames[3001:]:
    #Update while loading the data
    count += 1
    np_array = np.load(filename)
    
    #Padding the array
    np_zeros = np.zeros(MAXLEN - len(np_array))
    np_array = np.append(np_array, np_zeros)
    np_array = np.array([[np_array]])
    
    if count == 1:
      data = np_array
      print(data)
    else:
      data = np.vstack((data, np_array))
    if(count % (int)(len(filenames)/20) == 0):
      print("Loaded  " + str(count) + " files and it took " + str(time.time() - t) + " second for this iteration")
      total_t += time.time() - t
      t = time.time()
      
  #print("Normalizing")
  #data = normalizeAudio(data)
  print(data.shape)
  print("Done, took total of " + str(total_t) + " seconds")
  #np.savetxt('/content/gdrive/Team Drives/Neural Network/TIMIT/wav1', data, fmt='%d')
  np.save('/content/gdrive/Team Drives/Neural Network/TIMIT/wav2', data)
  return data


'''
  This function loads 3D numpy array of text and wav, and return them
'''
def load_data():
  now = time.time()
  filename = '/content/gdrive/Team Drives/Neural Network/TIMIT/text.npy'
  text = np.load(filename)
  #text = np.array([text])
  filename = '/content/gdrive/Team Drives/Neural Network/TIMIT/wav1.npy'
  wav = np.load(filename)
  print("Done, took " + str(time.time() - now) + " seconds to load")
  return text, wav



  '''
  This cell loads the dataset to text_data and wav_data.
  It also set the input and output size for neural network as "input_len" and "output_len"
'''

text_data, wav_data = load_data()
print(text_data.shape)
print(wav_data.shape)

#Only use first 3000 because you can only load 3000 wav files at a time
text_data = text_data[:3000]
print(text_data.shape)

input_len = len(text_data[0][0])
output_len = len(wav_data[0][0])

print(input_len)
print(output_len)
'''
  This cell loads a pretraiend model from google drive
'''
model_name = 'my_model.h5'
model = load_model('/content/gdrive/Team Drives/Neural Network/' + model_name)
'''
  This cell generates wav file for "input_string" (Pretty much our final product)
'''

module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
embed = hub.Module(module_url)
text_list = list()

#This is the text the model is going to try to speak
input_string = "Hi this is Alex, I am a dumbass. hello agian."
text_list.append(input_string)

with tf.Session() as session:
  session.run([tf.global_variables_initializer(), tf.tables_initializer()])
  message_embeddings = session.run(embed(text_list))
  
text_data = np.array([[message_embeddings[0]]])
print(text_data.shape)

wav = model.predict(text_data)
print(wav.shape)
print(wav[0][0])
rate = 16000
sp.io.wavfile.write('/content/gdrive/Team Drives/Neural Network/TIMIT/hi2.wav', rate, wav[0][0])

'''
  This cell creates a simple NN model, train, and evaluate it.
  Original Code from https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
'''
# create model

from keras.models import Sequential
from keras.layers import Dense

#Create and design model
model = Sequential()
model.add(Dense(512, input_shape=(1, input_len), activation='tanh'))
model.add(Dense(256, activation='tanh'))
model.add(Dense(256, activation = 'tanh'))
model.add(Dense(output_len, activation='tanh'))

# Compile model
model.compile(loss='huber', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(text_data, wav_data, epochs=100, batch_size=10)

# evaluate the model
scores = model.evaluate(text_data, wav_data)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))     

#Saving the model
model_name = 'my_model.h5'
model.save('/content/gdrive/Team Drives/Neural Network/my_model.h5')  # creates a HDF5 file 'my_model.h5'
