import sys
sys.path.append('/opt/cocoapi/PythonAPI')
import nltk
import os
import torch
import torch.utils.data as data
from vocabulary import Vocabulary
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
from tqdm import tqdm
import random
import json

def get_val_loader(transform,
               batch_size=1,
               vocab_threshold=None,
               vocab_file='./vocab.pkl',
               start_word="<start>",
               end_word="<end>",
               unk_word="<unk>",
               vocab_from_file=True,
               num_workers=0,
               cocoapi_loc='/opt'):
    """Returns the data loader.
    Args:
      transform: Image transform.
      batch_size: Batch size (if in testing mode, must have batch_size=1).
      vocab_threshold: Minimum word count threshold.
      vocab_file: File containing the vocabulary. 
      start_word: Special word denoting sentence start.
      end_word: Special word denoting sentence end.
      unk_word: Special word denoting unknown words.
      vocab_from_file: If False, create vocab from scratch & override any existing vocab_file.
                       If True, load vocab from from existing vocab_file, if it exists.
      num_workers: Number of subprocesses to use for data loading 
      cocoapi_loc: The location of the folder containing the COCO API: https://github.com/cocodataset/cocoapi
    """
    
    assert batch_size==1, "Please change batch_size to 1 if testing your model."
    assert os.path.exists(vocab_file), "Must first generate vocab.pkl from training data."
    assert vocab_from_file==True, "Change vocab_from_file to True."
    img_folder = os.path.join(cocoapi_loc, 'cocoapi/images/val2014/')
    annotations_file = os.path.join(cocoapi_loc, 'cocoapi/annotations/captions_val2014.json')
    mode='val',

    # COCO caption dataset.
    dataset = CoCoValDataset(transform=transform,
                          mode=mode,
                          batch_size=batch_size,
                          vocab_threshold=vocab_threshold,
                          vocab_file=vocab_file,
                          start_word=start_word,
                          end_word=end_word,
                          unk_word=unk_word,
                          annotations_file=annotations_file,
                          vocab_from_file=vocab_from_file,
                          img_folder=img_folder)

    data_loader = data.DataLoader(dataset=dataset,
                                    batch_size=dataset.batch_size,
                                    shuffle=True,
                                    num_workers=num_workers)

    return data_loader

class CoCoValDataset(data.Dataset):
    
    def __init__(self, transform, mode, batch_size, vocab_threshold, vocab_file, start_word, 
        end_word, unk_word, annotations_file, vocab_from_file, img_folder):
        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size
        self.vocab = Vocabulary(vocab_threshold, vocab_file, start_word,
            end_word, unk_word, annotations_file, vocab_from_file)
        self.img_folder = img_folder
        self.coco = COCO(annotations_file)
        self.coco_new_format = {}
        for ann in self.coco.anns:
            img_id = self.coco.anns[ann]['image_id']
            if img_id not in self.coco_new_format.keys():
                self.coco_new_format[img_id] = {}
                self.coco_new_format[img_id]["id"] = self.coco.anns[ann]['id']
                self.coco_new_format[img_id]["image_id"] = img_id
                self.coco_new_format[img_id]['captions'] = [self.coco.anns[ann]['caption']] 
            else:
                caption = self.coco.anns[ann]['caption']
                self.coco_new_format[img_id]['captions'].append(caption)
        
        self.ids = list(self.coco_new_format.keys())
        
    def __getitem__(self, index):
    
        # obtain image if in test mode
        img_id = self.ids[index]
        captions = self.coco_new_format[img_id]['captions']
        path = self.coco.loadImgs(img_id)[0]['file_name']

        # Convert image to tensor and pre-process using transform
        PIL_image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
        orig_image = np.array(PIL_image)
        image = self.transform(PIL_image)

        # return original image and pre-processed image tensor, language caption, tokenize caption
        return orig_image, image, captions, img_id

    def __len__(self):
        return len(self.ids)