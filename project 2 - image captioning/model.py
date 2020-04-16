import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet50


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        # resnet50 outperforms resnet152 with less layers
        # all layers apart from last pool and FC (output: 7x7x2048)
        resnet = resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        for param in self.resnet.parameters():
            param.requires_grad_(False)
        
        # add a FC = weight matrix for embeddings (2048 -> embed_size)
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)
        
        # output of forward will be of dim: (batch_size x 49 x embed_size)
        self.feature_dim = embed_size

    def forward(self, images):
        # don't finetune the pretrained layers, no need for gradients -> FASTER
        with torch.no_grad():
            features = self.resnet(images)
        
        # flatten each feature map (7x7x2048 -> 49x2048)
        features = features.view(features.size(0), features.size(1), -1)

        # reshape (batch x 49 x 2048)
        features = features.permute(0, 2, 1)

        # produce context matrix (batch x 49 x 512)
        features = self.fc(features)
        
        return features

    
class Attention(nn.Module):
    def __init__(self, feature_dim, hidden_dim, embed_size):
        super(Attention, self).__init__()
        self.embed_size = embed_size
        self.U = nn.Linear(hidden_dim, self.embed_size)
        self.W = nn.Linear(feature_dim, self.embed_size)
        self.V = nn.Linear(self.embed_size, 1)
        self.acti = nn.ReLU() # originally tanh in Bahdanau scoring
        self.softmax = nn.Softmax(1)
    
    def forward(self, feature_vectors, hidden_state):
        # have to unsqueeze dim 1 as hidden is of dim (batch,hiddendim)
        hidden_state = hidden_state.unsqueeze(1)
        U_h = self.U(hidden_state) # dim: batch, 1, embed_size
        W_f = self.W(feature_vectors) # dim: batch, 49, embed_size
        attention_scores_raw = self.acti(W_f + U_h)
        attention_scores = self.V(attention_scores_raw)
        attention_weights = self.softmax(attention_scores)
        
        context_matrix = attention_weights * feature_vectors
        context_vector = context_matrix.sum(1)
        
        return context_vector

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, attention, device):
        super(DecoderRNN, self).__init__()
        
        # embedding layer that turns tokens into a vector of a specified size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.attention = attention
        self.teacher_forcing = False
        self.device = device
        
        self.fc_init_hidden = nn.Linear(embed_size, hidden_size)
        self.fc_init_output = nn.Linear(embed_size, hidden_size)
        self.tanh = nn.Tanh()
        
        # GRU perform slightly better than LSTM for image captioning
        # -- An Empirical Study of Language CNN for Image Captioning, Gu et al, 2017
        # -- https://arxiv.org/pdf/1612.07086.pdf
        # they also tend to train faster and have less parameters
        self.gru = nn.GRU(input_size=embed_size*2, hidden_size=hidden_size, batch_first=False)
        
        # add a FC = produces scores for each token in the vocab, to then generate the output
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.Softmax
        
        # teacher forcing as described in section 10.2.1 of Deep Learning, Courville Goodfellow Bengio, 2015
        # https://www.deeplearningbook.org/contents/rnn.html
        self.teacher_forcing = True
        
        # Scoring embedded feature vectors (from encoder)
        

    def init_hidden_state(self, features):
        avg_features = features.mean(dim=1)
        
        hidden = self.fc_init_hidden(avg_features)
        hidden = self.tanh(hidden)
        
        output = self.fc_init_output(avg_features)
        output = self.tanh(output)
        
        return hidden, output
    
    
    def forward(self, features, captions):
        # indices and loop conditions
        batch_size = features.shape[0]
        captions_length = [len(caption) for caption in captions]
        max_caption_length = max(captions_length)
        
        # init tensors
        hidden, output = self.init_hidden_state(features)
        embeddings = self.word_embeddings(captions)
        predicted_captions = torch.zeros(batch_size, max_caption_length, self.vocab_size)
        
        # iterations (1-caption length)
        for i in range(max_caption_length):
            # attention
            context_vector = self.attention(features, hidden)
            
            # teacher forcing = use ground truth as previous output
            if i > 0 and self.teacher_forcing:
                embed = embeddings[:, i-1, :]
            else:
                embed = output
            
            # concat past prediction and attention context vector
            gru_input = torch.cat([embed, context_vector], dim=1)
            
            # takes a tensor of shape (batch_size, seq_len, input_size)
            hidden, output = self.gru(gru_input.unsqueeze(0), hidden.unsqueeze(0))
            
            # reshape and preapre output
            hidden, output = hidden.squeeze(0), output.squeeze(0)
            pred = self.fc_out(output)
            predicted_captions[:, i, :] = pred

        
        return predicted_captions

    def extract_word_index(self, pred):
        # extract index of embedding with highest probability
        _, word_index = pred[0].max(0)
        
        # remove gradient tracking
        word_index = word_index.detach()
        
        # remove from gpu
        if self.device != 'cpu':
            word_index = word_index.cpu()
        
        # get int value
        word_index = word_index.tolist()
        
        return word_index

    
    def sample(self, features, states=None, max_caption_length=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        batch_size = 1
        
        hidden, output = self.init_hidden_state(features)
        #output = states
        predicted_captions = []
        
        # first token should be <start>, which is 0
        #predicted_captions.append(self.word_embeddings(0))
        
        # iterations (1-caption length)
        for i in range(max_caption_length):
            context_vector = self.attention(features, hidden)
            gru_input = torch.cat([output, context_vector], dim=1)
            # takes a tensor of shape (batch_size, seq_len, input_size)
            hidden, output = self.gru(gru_input.unsqueeze(0), output.unsqueeze(0))
            hidden, output = hidden.squeeze(0), output.squeeze(0)

            pred = self.fc_out(output)
            word_index = self.extract_word_index(pred)
            predicted_captions.append(word_index)
        
        return predicted_captions
            
            
            
            