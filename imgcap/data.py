import os
import numpy as np
import pandas as pd
import spacy

import torch
import torch.nn as nn
from torch.utils import data


from sklearn import model_selection

import PIL
from PIL import Image



# spacy_eng = spacy.load('en_core_web_sm')

class Vocabulary(object):
    def __init__(self, freq_threshold: int, spacy_eng=None):
        self.itos: dict = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi: dict = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold
        if spacy_eng==None:
            self.spacy_eng = spacy.load('en_core_web_sm')
        else:
            self.spacy_eng = spacy_eng

    def __len__(self):
        return len(self.itos)

    def tokenizer_eng(self, text):
        return [tok.text.lower() for tok in self.spacy_eng.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1

                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]


class FlickrDataset(data.Dataset):
    def __init__(self, root_dir, caption_file, transform=None, freq_threshold=5,
                 train=True, split_val=0.2):
        self.root_dir = root_dir
        self.caption_file = caption_file
        self.df = pd.read_csv(caption_file)
        self.transform = transform
        
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.df['caption'].tolist())
        
        self.train = train 
        self.split_val = split_val
        self._do_split_train_valid()
        
#         # Get img, caption columns
#         self.imgs = self.df["image"]
#         self.captions = self.df["caption"]

        # Initialize vocabulary and build vocab

        
    def _do_split_train_valid(self):
        imgs_train, imgs_valid, caps_train, caps_valid = model_selection.train_test_split(
            self.df["image"], self.df["caption"], 
            test_size=self.split_val, random_state=16
        )
        
        if self.train:
            self.imgs = imgs_train
            self.captions = caps_train
        else:
            self.imgs = imgs_valid
            self.captions = caps_valid
            
        self.imgs = self.imgs.tolist()
        self.captions = self.captions.tolist()
        

    def __len__(self):
        return len(self.imgs)
    
    def _numericalized_caption(self, caption):
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])
        
        return numericalized_caption

    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.imgs[index]
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        ncaption = self._numericalized_caption(caption)

        return img, torch.tensor(ncaption)


class CaptionCollate:
    def __init__(self, pad_idx, batch_first=True):
        self.pad_idx = pad_idx
        self.batch_first = batch_first

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = nn.utils.rnn.pad_sequence(targets, batch_first=self.batch_first, 
                               padding_value=self.pad_idx)

        return imgs, targets
    

def flickr8k_dataloader(root_folder, caption_file, transform, train=True,
                    batch_size=32, num_workers=8, shuffle=True, pin_memory=True):

    dataset = FlickrDataset(root_folder, caption_file, transform=transform, train=train)
    pad_idx = dataset.vocab.stoi["<PAD>"]
    dataloader = data.DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers,
                            shuffle=shuffle, pin_memory=pin_memory, 
                            collate_fn=CaptionCollate(pad_idx=pad_idx))

    return dataloader, dataset