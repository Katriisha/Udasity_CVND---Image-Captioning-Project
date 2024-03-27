import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(weights=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.lstm = nn.LSTM(embed_size, hidden_size,num_layers=num_layers, batch_first = True)
        self.word_scores = nn.Linear(hidden_size, vocab_size)
        self.words_embed = nn.Embedding(vocab_size, embed_size)
         
    def forward(self, features, captions):
        captions_embed = self.words_embed(captions[:, :-1])
        lstm_input = torch.cat((features.unsqueeze(1), captions_embed), 1)
        
        out, _ = self.lstm(lstm_input)
        out = self.word_scores(out)
        return out

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        out, states = self.lstm(inputs, states)
        word_scores = self.word_scores(out)
        first_symbol = torch.argmax(word_scores).view(1, 1)
        embed_out = self.words_embed(first_symbol)

        description = []

        for i in range(max_len - 1):
            out, states = self.lstm(embed_out, states)
            word_scores = self.word_scores(out)
            symbol = torch.argmax(word_scores).view(1, 1)
            embed_out = self.words_embed(symbol)
            symbol_cpu = symbol.cpu().numpy()[0].item()
            if symbol_cpu == 1:
                break
            description.append(symbol_cpu)
            
        return description
        
         