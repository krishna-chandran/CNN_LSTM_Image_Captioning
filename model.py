import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet18(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.batnorm = nn.BatchNorm1d(embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        features = self.batnorm(features)
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=3, drop_prob=0.3):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(drop_prob)
        self.init_weights()
    
    def init_weights(self):
        ''' Initialize weights for fully connected layer and lstm forget gate bias'''
        
        # Set bias tensor to all 0.01
        self.linear.bias.data.fill_(0.01)
        # FC weights as xavier normal
        torch.nn.init.xavier_normal_(self.linear.weight)
        
        # init forget gate bias to 1
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n//4, n//2
                bias.data[start:end].fill_(1.)
        
    
    def forward(self, features, captions, lengths=None):
        """Decode image feature vectors and generates captions."""
        captions = captions[:,:-1]
        embeddings = self.embed(captions)
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.lstm(inputs)
        hiddens = self.dropout(hiddens)
        outputs = self.linear(hiddens)
        return outputs

    
    def sample(self, features, states=None):
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
#         inputs = features.unsqueeze(1)
        for i in range(20):                                    # maximum sampling length
            hiddens, states = self.lstm(features, states)        # (batch_size, 1, hidden_size), 
            outputs = self.linear(hiddens.squeeze(1))          # (batch_size, vocab_size)
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted.item())
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)                       # (batch_size, 1, embed_size)
#         sampled_ids = torch.cat(sampled_ids, 1)                # (batch_size, 20)
        return sampled_ids

