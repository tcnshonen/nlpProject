import torch
import torch.nn as nn

from .layers import ConvLayer, LinearLayer
from utils.constants import word_to_ix, ram_size


class Autoencoder(nn.Module):
    def __init__(self, embedding_dim=256, sentence_num=10, lstm_dim=128,
                 input_shape=(3, 224, 160), dropout=False, activation_name='relu'):
        super(Autoencoder, self).__init__()
        self.sentence_num = sentence_num
        self.input_shape = input_shape
        self.lstm_dim = lstm_dim
        self.config = {'dropout': dropout, 'activation_name': activation_name}

        self.encoder_config = {'dropout': dropout, 'activation_name': 'leakyrelu'}

        #Encoder
        self.encoder = nn.Sequential(
            ConvLayer(3, 64, 4, stride=2, padding=1, **self.encoder_config),
            ConvLayer(64, 128, 4, stride=2, padding=1, **self.encoder_config),
            ConvLayer(128, 256, 4, stride=2, padding=1, **self.encoder_config),
            ConvLayer(256, 512, 4, stride=2, padding=1, **self.encoder_config),
            ConvLayer(512, 1024, 4, stride=2, padding=1, **self.encoder_config)
        )

        #Flatten
        self.last_shape, self.flatten_num = self.flatten_shape(self.input_shape)
        self.fc1 = LinearLayer(self.flatten_num, 256, **self.config)

        #Memory Decoder
        self.fc_mem = LinearLayer(256, ram_size * 4, **self.config)
        self.memory_decoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.BatchNorm1d(ram_size),
            nn.ELU(),
            nn.Dropout(),
            nn.Linear(64, 256),
        )

        #Image Decoder
        self.fc_img = LinearLayer(256, self.flatten_num, **self.config)
        self.image_decoder = nn.Sequential(
            ConvLayer(1024, 512, 4, stride=2, padding=1, transpose=True, **self.config),
            ConvLayer(512, 256, 4, stride=2, padding=1, transpose=True, **self.config),
            ConvLayer(256, 128, 4, stride=2, padding=1, transpose=True, **self.config),
            ConvLayer(128, 64, 4, stride=2, padding=1, transpose=True, **self.config),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Tanh(),
        )

        #RNN
        self.word_embeddings = nn.Embedding(len(word_to_ix)-1, 64)
        self.lstm = nn.LSTM(64, self.lstm_dim, batch_first=True)
        self.fc_sen = nn.Linear(self.lstm_dim, 256)


    def flatten_shape(self, shape):
        temp = torch.rand(1, *shape)
        temp = self.encoder(temp)
        temp_shape = temp.size()
        temp_num = temp.data.view(1, -1).size(1)

        return temp_shape[1:], temp_num

    def flatten_forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, self.flatten_num)
        x = self.fc1(x)

        return x

    def get_diff(self, x1, x2):
        x1 = self.flatten_forward(x1)
        x2 = self.flatten_forward(x2)
        x_diff = x2 - x1

        return x1, x2, x_diff

    def decoder_forward(self, x):
        img_x = self.fc_img(x)
        img_x = img_x.view(-1, *self.last_shape)
        img_x = self.image_decoder(img_x)

        mem_x = self.fc_mem(x)
        mem_x = mem_x.view(-1, ram_size, 4)
        mem_x = self.memory_decoder(mem_x)
        mem_x.transpose_(1, 2)

        return img_x, mem_x

    def forward(self, x1, x2, sentence):
        #Encoder
        x1, x2, x_diff = self.get_diff(x1, x2)

        #Decoder
        img_x1, mem_x1 = self.decoder_forward(x1)
        img_x2, mem_x2 = self.decoder_forward(x2)

        #Sentence
        embedding = self.word_embeddings(sentence)
        _, (embedding, _) = self.lstm(embedding)
        embedding = embedding.view(-1, self.lstm_dim)
        embedding = self.fc_sen(embedding)

        return img_x1, mem_x1, img_x2, mem_x2, x_diff, embedding
