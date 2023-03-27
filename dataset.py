import numpy as np
import torch
torch.manual_seed(1010)
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import ViTFeatureExtractor, AutoTokenizer, BridgeTowerProcessor
from PIL import Image


class TwitterDataset(Dataset):
    def __init__(self,args,train):
        super(TwitterDataset, self).__init__()
        self.path = args.path
        self.args = args
        self.train = train
        self.pad_id = args.pad_id
        self.patch_len = args.patch_len
        self.attention_mode = args.attention_mode
        self.id2label = {
            "0":"NEG",
            "1":"NEU",
            "2":"POS"
        }
        self.label2id = {
            "O":0,
            "B-POS":1,
            "B-NEG":2,
            "B-NEU":3,
            "I":4
        }
        # self.tokenizer = AutoTokenizer.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc",add_prefix_space=True)
        self.extractor = BridgeTowerProcessor.from_pretrained(self.args.pretrained_model,add_prefix_space=True)
        text,label,image_ids = self.preprocessing()
        self.input_ids,self.labels,self.images = self.process(text,label,image_ids)
        self.label_pad = len(self.label2id)


    def __len__(self):
        assert len(self.input_ids) == len(self.labels)
        return len(self.input_ids)


    def __getitem__(self, item):
        return self.input_ids[item],self.labels[item],self.images[item]


    def collate_fn(self,batch):
        images = []
        labels = []
        input_ids = []
        for text,label,image in batch:
            input_ids.append(text)
            labels.append(label)
            images.append(image)

        tokenized_inputs = self.extractor(images=images,text=input_ids, truncation=True, is_split_into_words=True,
                                          padding='max_length', max_length=self.patch_len,return_tensors="pt")

        text_labels = []

        for i, v in enumerate(labels):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            label_ids = []
            pre_word_idx = None
            for word_idx in word_ids:  # Set the special tokens to 5.
                if word_idx is None:
                    label_ids.append(self.pad_id)
                else:
                    if pre_word_idx != word_idx:
                        label_ids.append(v[word_idx])
                    else:
                        label_ids.append(self.pad_id)
                pre_word_idx = word_idx

            if self.attention_mode == "concat_attention":
                text_labels.append(label_ids + [self.pad_id * self.patch_len])
            else:
                text_labels.append(label_ids)

        tokenized_inputs["labels"] = torch.tensor(text_labels)
        return tokenized_inputs['input_ids'],tokenized_inputs['attention_mask'],tokenized_inputs['pixel_values'],tokenized_inputs['pixel_mask'],tokenized_inputs['labels']


    def process(self,text,label,image_ids):
        input_ids = []
        labels = []
        images = []
        path = r"{0}/images/{1}_images/".format(self.path,self.args.dataset.lower())
        for i in range(len(text)):
            input_ids.append(text[i])
            temp = []
            for t in label[i]:
                temp.append(self.label2id[t])
            labels.append(temp)
            image_path = path + image_ids[i]
            images.append(Image.open(image_path).convert("RGB"))
        return input_ids,labels,images


    def preprocessing(self):
        if self.train:
            data = pd.read_csv("{0}/{1}/train.tsv".format(self.path,self.args.dataset.lower()),sep='\t')
        else:
            data = pd.read_csv("{0}/{1}/test.tsv".format(self.path,self.args.dataset.lower()),sep='\t')
        text = []
        label = []
        image_ids = []
        for i in range(len(data)):
            if data['#2 ImageID'][i] not in image_ids:
                image_ids.append(data['#2 ImageID'][i])
                index = data['#3 String'][i].split().index("$T$")
                replace_len = len(data['#3 String.1'][i].split())
                replace_word = data['#3 String.1'][i]
                text.append(data['#3 String'][i].replace("$T$",replace_word).split())

                temp_label = ["O"] * len(data['#3 String'][i].replace("$T$",replace_word).split())
                temp_label[index] = "B-"+str(self.id2label[str(data['#1 Label'][i])])
                temp_label[index+1:index+replace_len] = ["I"] * (replace_len - 1)
                label.append(temp_label)
            else:
                index = image_ids.index(data['#2 ImageID'][i])
                replace_word = data['#3 String.1'][i].split()
                replace_index = -1
                for j in range(len(text[index])-len(replace_word)+1):
                    if text[index][j:j+len(replace_word)] == replace_word:
                        replace_index = j
                        break

                if replace_index == -1:
                    raise AttributeError("cannot find the location")
                replace_len = len(replace_word)
                temp_label = label[index]
                temp_label[replace_index] = "B-"+str(self.id2label[str(data['#1 Label'][i])])
                temp_label[replace_index+1:replace_index+replace_len] = ["I"] * (replace_len-1)
                label[index] = temp_label
                assert len(temp_label) == len(text[-1])

        return text, label, image_ids

