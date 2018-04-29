from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import nltk

import pickle
import random

padToken, goToken, eosToken, unknownToken = 0, 1, 2, 3

class Batch:
    #batch class,which contains encoder_input,decoder_input,decoder_label,decoder_sample length
    def __init__(self):
        self.encoderSeqs = []
        self.decoderSeqs = []
        self.targetSeqs = []
        self.weights = []

def loadDataset(filename):
    '''
    load sample data
    :param filename: file path,which is a dictionary,including word2id、id2word,trainingSamples
    :return: word2id, id2word, trainingSamples
    '''
    dataset_path = os.path.join(filename)
    print('Loading dataset from {}'.format(dataset_path))
    with open(dataset_path, 'rb') as handle:
        data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
        word2id = data['word2id']
        id2word = data['id2word']
        trainingSamples = data['trainingSamples']
    return word2id, id2word, trainingSamples

def createBatch(samples, en_de_seq_len):
    '''
    According to samples(batch class data),do padding and constructing the stucture that placeholder need
    :param samples: a batch class data,list,every element is [question, answer]
    :param en_de_seq_len: list,the first element expresses the maximum length of source end order,the second one expresses the maximum length of target end order
    :return: a format which could input to feed_dict directly after being processed.
    '''
    batch = Batch()
    #get batch size accoding to samples lengths.
    batchSize = len(samples)
    #input the quetions and answers of each data to the corresponding variables respectively.
    for i in range(batchSize):
        sample = samples[i]
        batch.encoderSeqs.append(list(reversed(sample[0])))  # the modle could be improved by reversing inputs.
        batch.decoderSeqs.append([goToken] + sample[1] + [eosToken])  # Add the <go> and <eos> tokens
        batch.targetSeqs.append(batch.decoderSeqs[-1][1:])  # Same as decoder, but shifted to the left (ignore the <go>)
        # padding every element to designated length and constructing masks of weights order.
        batch.encoderSeqs[i] = [padToken] * (en_de_seq_len[0] - len(batch.encoderSeqs[i])) + batch.encoderSeqs[i]
        batch.weights.append([1.0] * len(batch.targetSeqs[i]) + [0.0] * (en_de_seq_len[1] - len(batch.targetSeqs[i])))
        batch.decoderSeqs[i] = batch.decoderSeqs[i] + [padToken] * (en_de_seq_len[1] - len(batch.decoderSeqs[i]))
        batch.targetSeqs[i] = batch.targetSeqs[i] + [padToken] * (en_de_seq_len[1] - len(batch.targetSeqs[i]))

    #--------------------reshape data into order_length*batch_size structure------------------------
    encoderSeqsT = []  # Corrected orientation
    for i in range(en_de_seq_len[0]):
        encoderSeqT = []
        for j in range(batchSize):
            encoderSeqT.append(batch.encoderSeqs[j][i])
        encoderSeqsT.append(encoderSeqT)
    batch.encoderSeqs = encoderSeqsT

    decoderSeqsT = []
    targetSeqsT = []
    weightsT = []
    for i in range(en_de_seq_len[1]):
        decoderSeqT = []
        targetSeqT = []
        weightT = []
        for j in range(batchSize):
            decoderSeqT.append(batch.decoderSeqs[j][i])
            targetSeqT.append(batch.targetSeqs[j][i])
            weightT.append(batch.weights[j][i])
        decoderSeqsT.append(decoderSeqT)
        targetSeqsT.append(targetSeqT)
        weightsT.append(weightT)
    batch.decoderSeqs = decoderSeqsT
    batch.targetSeqs = targetSeqsT
    batch.weights = weightsT

    return batch

def getBatches(data, batch_size, en_de_seq_len):
    '''
    Divide original data into diiferent small batch size accoding to the whole reading data and batch_size. Call createBatch function to process samples. 
    :param data: the trainingSamples after being processed by loadDataset function.
    :param batch_size
    :param en_de_seq_len: list,the first element expresses the maximum length of source end order,the second one expresses the maximum length of target end order
    :return: a list which contains batch size samples sa elements and could input to feed_dict directly after being processed.
    '''
    #shuffle samples before each epoch.
    random.shuffle(data)
    batches = []
    data_len = len(data)
    def genNextSamples():
        for i in range(0, data_len, batch_size):
            yield data[i:min(i + batch_size, data_len)]

    for samples in genNextSamples():
        batch = createBatch(samples, en_de_seq_len)
        batches.append(batch)
    return batches

def sentence2enco(sentence, word2id, en_de_seq_len):
    '''
    Transform inputing sentences by user to data which could feed into modle directly, then change sentences into id and call createBatch function.
    :param sentence: inputing sentences 
    :param word2id: a directionary of corresponding relationship between word and id
    :param en_de_seq_len: list,the first element expresses the maximum length of source end order,the second one expresses the maximum length of target end order
    :return: processing data which could feed into modle directly for predicting
    '''
    if sentence == '':
        return None
    #word segmentation
    tokens = nltk.word_tokenize(sentence)
    if len(tokens) > en_de_seq_len[0]:
        return None
    #transform every word to id.
    wordIds = []
    for token in tokens:
        wordIds.append(word2id.get(token, unknownToken))
    #call createBatch fuction to constructing batch.
    batch = createBatch([[wordIds, []]], en_de_seq_len)
    return batch
