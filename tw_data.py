import sugartensor as tf
import numpy as np
import pandas as pd
import preprocess_raw_data
import os
from os.path import join
import pickle
DATA_DIR = 'data'


def load_corpus_dict_test(path,b2i,min_len,max_len):
    all_seqs_df = pd.read_csv(path, sep=preprocess_raw_data.SRC_TRG_SEP)
    # make character-level parallel corpus
    all_byte, sources = [], [],
    for i,row in all_seqs_df.iterrows():
        print ('\rrow %d' % (i)),
        try:
            src = [ord(ch) for ch in row['src_seq']]  # source language byte stream
            sources.append(src)
            all_byte.extend(src)
        except:
            sources.append('')

    # remove short and long sentence
    src = []
    for s in sources:
        if min_len <= len(s) < max_len:
            src.append(s)
        else:
            src.append('')

    # convert to index list and add <EOS> to end of sentence
    for i in range(len(src)):
        print ('\rprocessing source %d' % (i)),
        src[i] = [b2i[ch] for ch in src[i]] + [1]

    # zero-padding
    for i in range(len(src)):
        print ('\rprocessing target %d' % (i)),
        src[i] += [0] * (max_len - len(src[i]))

    users  = list(all_seqs_df['user'])
    ids = list(all_seqs_df['id'])
    return src,users,ids

def load_corpus_dict(path,b2i,min_len,max_len):
    all_seqs_df = pd.read_csv(path, sep=preprocess_raw_data.SRC_TRG_SEP).dropna()
    # make character-level parallel corpus
    all_byte, sources, targets = [], [], []
    for i,row in all_seqs_df.iterrows():
        try:
            print ('\rrow %d' % (i)),
            src = [ord(ch) for ch in row[0]]  # source language byte stream
            tgt = [ord(ch) for ch in row[1]]  # target language byte stream
            sources.append(src)
            targets.append(tgt)
            all_byte.extend(src + tgt)
        except:
            continue


    # remove short and long sentence
    src, tgt = [], []
    for s, t in zip(sources, targets):
        if min_len <= len(s) < max_len and min_len <= len(t) < max_len:
            src.append(s)
            tgt.append(t)

    # convert to index list and add <EOS> to end of sentence
    for i in range(len(src)):
        print ('\rprocessing source %d' % (i)),
        src[i] = [b2i[ch] for ch in src[i]] + [1]
        tgt[i] = [b2i[ch] for ch in tgt[i]] + [1]

    # zero-padding
    for i in range(len(tgt)):
        print ('\rprocessing target %d' % (i)),
        src[i] += [0] * (max_len - len(src[i]))
        tgt[i] += [0] * (max_len - len(tgt[i]))

    # swap source and target : french -> english
    return src, tgt
    
def to_batch_dict(self, sentences, b2i):

    # convert to index list and add <EOS> to end of sentence
    for i in range(len(sentences)):
        sentences[i] = [b2i[ord(ch)] for ch in sentences[i]] + [1]

    # zero-padding
    for i in range(len(sentences)):
        sentences[i] += [0] * (self.max_len - len(sentences[i]))

    return sentences
    

class TwitData(object):

    def __init__(self, batch_size=8, name='train', path=join(DATA_DIR,'out_data.data'),
                 b2i={}):
        if name == "train":
            
            print ("Loading corpus...")
            # load train corpus
            if not b2i:
                sources, targets = self._load_corpus(mode='train',path=path)
            else:
                sources, targets = load_corpus_dict(path,b2i,0,150)
            print ("Converting source to tensors...")
            # to constant tensor
            source = tf.convert_to_tensor(sources)
            print ("Converting target to tensors...")
            target = tf.convert_to_tensor(targets)
    
            # create queue from constant tensor
            source, target = tf.train.slice_input_producer([source, target])
    
            # create batch queue
            batch_queue = tf.train.shuffle_batch([source, target], batch_size,
                                                 num_threads=4, capacity=batch_size*8,
                                                 min_after_dequeue=batch_size*4, name=name)
    
            # split data
            self.source, self.target = batch_queue
    
            # calc total batch count
            self.num_batch = len(sources) // batch_size
    
            # print info
            tf.sg_info('Train data loaded.(total data=%d, total batch=%d)' % (len(sources), self.num_batch))
        
        if name == "test":
            print ("Loading test corpus...")
            
            sources = self._load_corpus(mode='test',path=path)
            targets = np.zeros_like(sources)
            print ("Converting source to tensors...")
            # to constant tensor
            source = tf.convert_to_tensor(sources)
            print ("Converting target to tensors...")
            target = tf.convert_to_tensor(targets)
    
            # create queue from constant tensor
            source, target = tf.train.slice_input_producer([source, target])
    
            # create batch queue
            batch_queue = tf.train.shuffle_batch([source, target], batch_size,
                                                 num_threads=4, capacity=batch_size*8,
                                                 min_after_dequeue=batch_size*4, name=name)
    
            # split data
            self.source, self.target = batch_queue
    
            # calc total batch count
            self.num_batch = len(sources) // batch_size
    
            # print info
            tf.sg_info('Train data loaded.(total data=%d, total batch=%d)' % (len(sources), self.num_batch))
        

#    def _load_corpus_dict()
    def _load_corpus(self, mode='train',path=join(DATA_DIR,'out_data_100000.data')):
        if mode == "train":
            
            all_seqs_df = pd.read_csv(path, sep=preprocess_raw_data.SRC_TRG_SEP).dropna()
            # make character-level parallel corpus
            all_byte, sources, targets = [], [], []
            for i,row in all_seqs_df.iterrows():
                print ('\rrow %d' % (i)),
                src = [ord(ch) for ch in row['src_seq']]  # source language byte stream
                tgt = [ord(ch) for ch in row['trg_seq']]  # target language byte stream
                sources.append(src)
                targets.append(tgt)
                all_byte.extend(src + tgt)
    
            # make vocabulary
            self.index2byte = [0, 1] + list(np.unique(all_byte))  # add <EMP>, <EOS> tokens
            self.byte2index = {}
            for i, b in enumerate(self.index2byte):
                self.byte2index[b] = i
            self.voca_size = len(self.index2byte)
            self.max_len = 150
            self.min_len = 0
    
            # remove short and long sentence
            src, tgt = [], []
            for s, t in zip(sources, targets):
                if self.min_len <= len(s) < self.max_len and self.min_len <= len(t) < self.max_len:
                    src.append(s)
                    tgt.append(t)
    
            # convert to index list and add <EOS> to end of sentence
            for i in range(len(src)):
                print ('\rprocessing source %d' % (i)),
                src[i] = [self.byte2index[ch] for ch in src[i]] + [1]
                tgt[i] = [self.byte2index[ch] for ch in tgt[i]] + [1]
    
            # zero-padding
            for i in range(len(tgt)):
                print ('\rprocessing target %d' % (i)),
                src[i] += [0] * (self.max_len - len(src[i]))
                tgt[i] += [0] * (self.max_len - len(tgt[i]))
    
            # swap source and target : french -> english
            return src, tgt
        
        if mode == "test":
            all_seqs_df = pd.read_csv(path, sep=preprocess_raw_data.SRC_TRG_SEP).dropna()
            # make character-level parallel corpus
            all_byte, sources = [], []
            for i,row in all_seqs_df.iterrows():
                print ('\rrow %d' % (i)),
                src = [ord(ch) for ch in row['src_seq']]  # source language byte stream
                sources.append(src)
                all_byte.extend(src)
    
            # make vocabulary
            self.index2byte = [0, 1] + list(np.unique(all_byte))  # add <EMP>, <EOS> tokens
            self.byte2index = {}
            for i, b in enumerate(self.index2byte):
                self.byte2index[b] = i
            self.voca_size = len(self.index2byte)
            self.max_len = 150
            self.min_len = 0
    
            # remove short and long sentence
            src, tgt = [], []
            for s, t in zip(sources, targets):
                if self.min_len <= len(s) < self.max_len and self.min_len <= len(t) < self.max_len:
                    src.append(s)
                    tgt.append(t)
    
            # convert to index list and add <EOS> to end of sentence
            for i in range(len(src)):
                print ('\rprocessing source %d' % (i)),
                src[i] = [self.byte2index[ch] for ch in src[i]] + [1]
                tgt[i] = [self.byte2index[ch] for ch in tgt[i]] + [1]
    
            # zero-padding
            for i in range(len(tgt)):
                print ('\rprocessing target %d' % (i)),
                src[i] += [0] * (self.max_len - len(src[i]))
                tgt[i] += [0] * (self.max_len - len(tgt[i]))
    
            # swap source and target : french -> english
            return src, tgt
            

    def to_batch(self, sentences):
        

        # convert to index list and add <EOS> to end of sentence
        for i in range(len(sentences)):
            sentences[i] = [self.byte2index[ord(ch)] for ch in sentences[i]] + [1]

        # zero-padding
        for i in range(len(sentences)):
            sentences[i] += [0] * (self.max_len - len(sentences[i]))

        return sentences

    def print_index(self, indices):
        for i, index in enumerate(indices):
            str_ = ''
            for ch in index:
                if ch > 1:
                    str_ += unichr(self.index2byte[ch])
                elif ch == 1:  # <EOS>
                    break
            print '[%d]' % i + str_

if __name__ == "__main__":
    
    twdata = TwitData()
#    sources,targets = twdata._load_corpus()
    
    b2i = twdata.byte2index
    i2b = twdata.index2byte
    with open('endec100k.pkl', 'wb') as f_out:
        pickle.dump(b2i,f_out)
        pickle.dump(i2b,f_out)
        