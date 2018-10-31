import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
train_on_gpu = True
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
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
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2, drop_prob=0.5):
        super(DecoderRNN, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.drop_prob = drop_prob
        
        self.caption_embeddings = nn.Embedding(vocab_size, embed_size)
        
        # define the LSTM
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
                            dropout=drop_prob, batch_first=True)
        
        # define a dropout layer
        self.dropout = nn.Dropout(drop_prob)
                            
        # define the final, fully-connected output layer
        self.fc = nn.Linear(hidden_size, vocab_size)
        
                            
        # initialize the weights
        #self.init_weights()
        
    
    def forward(self, features, captions):
        ''' Forward pass through the network '''
        # remove end token from captions
        captions = captions[:,:-1]
        
        # embedd captions
        caption_embeds = self.caption_embeddings(captions)
        
        # concatenate the feature and caption embeds
        inputs = torch.cat((features.unsqueeze(1),caption_embeds),1)
        
        # the first value returned by LSTM is all of the hidden states throughout
        # the sequence. the second is just the most recent hidden state
        out, hidden = self.lstm(inputs)
        
        # pass out through a droupout layer
        out = self.dropout(out)
                                
        # put out through the fully-connected layer
        out = self.fc(out)

        return out
        
    def init_weights(self):
        ''' Initialize weights for fully connected layer '''
        initrange = 0.1
        
        # Set bias tensor to all zeros
        self.fc.bias.data.fill_(0)
        # FC weights as random uniform
        self.fc.weight.data.uniform_(-1, 1)
        
        # init forget gate bias to 1
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n//4, n//2
                bias.data[start:end].fill_(1.)
        # "Importantly, adding a bias of size 1 significantly improved 
        # the performance of the LSTM on tasks where it fell behind the 
        # GRU and MUT1. Thus we recommend adding a bias of 1 to the forget 
        # gate of every LSTM in every application; it is easy to do often 
        # results in better performance on our tasks. This adjustment is 
        # the simple improvement over the LSTM that we set out to discover."
        # http://proceedings.mlr.press/v37/jozefowicz15.pdf
        # https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745/4
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        if (train_on_gpu):
            hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_().cuda(),
                  weight.new(self.num_layers, batch_size, self.hidden_size).zero_().cuda())
        else:
            hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_(),
                      weight.new(self.num_layers, batch_size, self.hidden_size).zero_())
        
        return hidden
    
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        pass