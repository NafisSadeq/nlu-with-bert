import numpy as np
import torch
import random
from nltk.tokenize import word_tokenize
import math
from collections import Counter, defaultdict


class Dataloader:
    def __init__(self, intent_vocab, tag_vocab, pretrained_weights='glove/glove.6B.300d.txt'):
        """
        :param intent_vocab: list of all intents
        :param tag_vocab: list of all tags
        """
        self.intent_vocab = intent_vocab
        self.tag_vocab = tag_vocab
        self.intent_dim = len(intent_vocab)
        self.tag_dim = len(tag_vocab)
        self.id2intent = dict([(i, x) for i, x in enumerate(intent_vocab)])
        self.intent2id = dict([(x, i) for i, x in enumerate(intent_vocab)])
        self.id2tag = dict([(i, x) for i, x in enumerate(tag_vocab)])
        self.tag2id = dict([(x, i) for i, x in enumerate(tag_vocab)])
        try:
            self.words = [l.split()[0] for l in open(pretrained_weights, 'r')]
        except:
            print("[ERROR] Please use `bash download_glove.sh` to download the pretrained embedding weights!")
            exit()
        self.word2id = defaultdict(lambda: len(self.words)+1)
        self.word2id.update(dict(zip(self.words, range(1, len(self.words)+1))))
        self.id2word = {self.word2id[w]:w for w in self.word2id}
        self.tokenizer = word_tokenize
        self.data = {}
        self.intent_weight = [1] * len(self.intent2id)

    def load_data(self, data, data_key, cut_sen_len, use_bert_tokenizer=False):
        """
        sample representation: [list of words, list of tags, list of intents, original dialog act]
        :param data_key: train/val/test
        :param data:
        :return:
        """
        self.data[data_key] = data
        max_sen_len, max_context_len = 0, 0
        sen_len = []
        context_len = []
        for d in self.data[data_key]:
            max_sen_len = max(max_sen_len, len(d[0]))
            sen_len.append(len(d[0]))
            # d = (tokens, tags, intents, da2triples(turn["dialog_act"], context(list of str))
            if cut_sen_len > 0:
                d[0] = d[0][:cut_sen_len]
                d[1] = d[1][:cut_sen_len]
                d[4] = [' '.join(s.split()[:cut_sen_len]) for s in d[4]]

            max_context_len = max(max_context_len, len(d[4]))
            context_len.append(len(d[4]))

            word_seq = d[0]
            tag_seq = d[1]
            new2ori = None
            d.append(new2ori)
            d.append(self.seq_word2id(word_seq))
            d.append(self.seq_tag2id(tag_seq))
            d.append(self.seq_intent2id(d[2]))
            # d = (tokens, tags, intents, da2triples(turn["dialog_act"]), context(token id), new2ori, new_word_seq, tag2id_seq, intent2id_seq)
            if data_key=='train':
                for intent_id in d[-1]:
                    self.intent_weight[intent_id] += 1
        if data_key == 'train':
            train_size = len(self.data['train'])
            for intent, intent_id in self.intent2id.items():
                neg_pos = (train_size - self.intent_weight[intent_id]) / self.intent_weight[intent_id]
                self.intent_weight[intent_id] = np.log10(neg_pos)
            self.intent_weight = torch.tensor(self.intent_weight)
        print('max sen bert len', max_sen_len)
        print(sorted(Counter(sen_len).items()))
        print('max context bert len', max_context_len)
        print(sorted(Counter(context_len).items()))

    def seq_tag2id(self, tags):
        return [self.tag2id[x] if x in self.tag2id else self.tag2id['O'] for x in tags]

    def seq_id2tag(self, ids):
        return [self.id2tag[x] for x in ids]

    def seq_intent2id(self, intents):
        return [self.intent2id[x] for x in intents if x in self.intent2id]

    def seq_id2intent(self, ids):
        return [self.id2intent[x] for x in ids]

    def seq_word2id(self, words):
        return [self.word2id[x] for x in words]

    def seq_id2word(self, ids):
        return [self.id2word[x] if x in self.word2id else '<UNK>' for x in ids]

    def pad_batch(self, batch_data):
        batch_size = len(batch_data)
        max_seq_len = max([len(x[-3]) for x in batch_data]) + 2
        word_mask_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        word_seq_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        tag_mask_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        tag_seq_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        intent_tensor = torch.zeros((batch_size, self.intent_dim), dtype=torch.float)
        context_max_seq_len = max([len(x[-5]) for x in batch_data])
        context_mask_tensor = torch.zeros((batch_size, context_max_seq_len), dtype=torch.long)
        context_seq_tensor = torch.zeros((batch_size, context_max_seq_len), dtype=torch.long)
        for i in range(batch_size):
            words = batch_data[i][-3]
            tags = batch_data[i][-2]
            intents = batch_data[i][-1]
            indexed_tokens = words
            sen_len = len(words)
            word_seq_tensor[i, :sen_len] = torch.LongTensor([indexed_tokens])
            tag_seq_tensor[i, :sen_len] = torch.LongTensor(tags)
            word_mask_tensor[i, :sen_len] = torch.LongTensor([1] * sen_len)
            tag_mask_tensor[i, :sen_len] = torch.LongTensor([1] * (sen_len))
            for j in intents:
                intent_tensor[i, j] = 1.
            context_len = len(batch_data[i][-5])
            context_seq_tensor[i, :context_len] = torch.LongTensor([1] * context_len) # TODO: add context
            context_mask_tensor[i, :context_len] = torch.LongTensor([1] * context_len)

        return word_seq_tensor, tag_seq_tensor, intent_tensor, word_mask_tensor, tag_mask_tensor, context_seq_tensor, context_mask_tensor

    def get_train_batch(self, batch_size):
        batch_data = random.choices(self.data['train'], k=batch_size)
        return self.pad_batch(batch_data)

    def yield_batches(self, batch_size, data_key):
        batch_num = math.ceil(len(self.data[data_key]) / batch_size)
        for i in range(batch_num):
            batch_data = self.data[data_key][i * batch_size:(i + 1) * batch_size]
            yield self.pad_batch(batch_data), batch_data, len(batch_data)
