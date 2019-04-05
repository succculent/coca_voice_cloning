import numpy
import tensorflow as tf
import tensorflow_hub as hub
import glob

def load_txt():
    ret = list()
    data_path = 'TIMIT/TRAIN/*/*/*.TXT'
    data = glob.glob(data_path)
    print(data[0:10]) #To see if its working
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
    embed = hub.Module(module_url)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer(), tf .tables_initialier())
        for file in data[0:10]:
            text_file = open(file, "r+")
            text = text_file.read()
            text_list = [text]
            message_embeddings = session.run(embed(text_list))
            print(message_embeddings)
            #print(text.read())


def load_wav(path):
    ret = list()
    data_path = 'TIMIT/TRAIN/*/*/*.WAV'
    data = glob.glob(data_path)
    print(data[0:10])
