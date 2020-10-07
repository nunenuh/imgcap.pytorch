import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
import matplotlib.pyplot as plt


class ImageCaptionTest(object):
    def __init__(self, model, vocab, max_len=20):
        self.model = model
        self.vocab = vocab
        self.max_len = max_len
        
        self.model.eval()
        self.encoder = model.encoder
        self.decoder = model.decoder
        
    def process_sentence_list(self, sentences):
        sentence_list = []
        for sentence in sentences:
            sentence_list.append(self.clean_sentence(sentence))
        return sentence_list
    
    def clean_sentence(self, sentence_index):
        sentence = ""
        for i in sentence_index:
            word = self.vocab.itos[i]
            if (word == self.vocab.start_word):
                continue
            elif (word == self.vocab.end_word):
                break
            else:
                sentence = sentence + " " + word
        return sentence
    
    def sample(self, images, states=None):
        with torch.no_grad():
            inputs = self.encoder(images).unsqueeze(dim=1)
            sampled_ids = []
            for i in range(self.max_len):
                hiddens, states = self.decoder.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
                outputs = self.decoder.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
                _, predicted = outputs.max(1)                        # predicted: (batch_size)
                sampled_ids.append(predicted)
                inputs = self.decoder.embed(predicted)                       # inputs: (batch_size, embed_size)
                inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
            sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
            return sampled_ids.tolist()
    
    def generate_caption(self, images):
        sentences_indexs = self.sample(images)
        sentences = self.process_sentence_list(sentences_indexs)
        return sentences
    
    def show_result(self, images, ground_truth):
        result = self.generate_caption(images)
        
        print(f'predicted    : {result[0]}')
        print(f'ground truth : {self.clean_sentence(ground_truth.tolist())}')
        plt.imshow(images.squeeze().permute(1,2,0), cmap='gray')
    
def clean_sentence(sentence_index, vocab):
    sentence = ""
    for i in sentence_index:
        word = vocab.itos[i]
        if (word == vocab.start_word):
            continue
        elif (word == vocab.end_word):
            break
        else:
            sentence = sentence + " " + word
    return sentence