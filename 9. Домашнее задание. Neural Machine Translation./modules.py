import random
import torch
from torch import nn
from torch.nn import functional as F

def softmax(x): # с tempreture=10, отвечает за гладкость
    e_x = torch.exp(x / 10)
    return e_x / torch.sum(e_x, dim=0)

# do we want to try different n_layers for encoder and decoder?
# maybe code that later? TODO

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout, bidirectional=False):
        super().__init__()
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, bidirectional=bidirectional, dropout=dropout, num_layers=n_layers)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        output, hidden = self.rnn(embedded)
        return output, hidden


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim, hidden_layers=1, enc_bidirectional=None):
        super().__init__()
        
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        
        self.attn = nn.Linear((enc_hid_dim*(1+enc_bidirectional) + dec_hid_dim)*hidden_layers, enc_hid_dim)
        self.v = nn.Linear(enc_hid_dim, 1)
        
    def forward(self, hidden, encoder_outputs):
        # TODO: does the gradient propagate through all the [tensor]*num ops?
        
        # encoder_outputs = [src sent len, batch size, enc_hid_dim]
        # hidden = [1, batch size, dec_hid_dim]
        
        # repeat hidden and concatenate it with encoder_outputs
        '''your code'''
        hidden = hidden.reshape(1,encoder_outputs.shape[1],-1)
        hidden_repeated = torch.cat([hidden]*encoder_outputs.shape[0])
        result = torch.cat((encoder_outputs, hidden_repeated),2)
        # calculate energy
        '''your code'''
        energy = torch.tanh(self.attn(result))
        # get attention, use softmax function which is defined, can change temperature
        '''your code'''
        attention = self.v(energy)
        return softmax(attention) #'''your code'''
    
    
class DecoderWithAttention(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention, n_layers=1):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.attention = attention;
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.GRU(input_size = emb_dim+enc_hid_dim, 
                          hidden_size = dec_hid_dim) # use GRU
        
        self.out = nn.Linear(dec_hid_dim+emb_dim+enc_hid_dim, output_dim) #'''your code''' # linear layer to get next word
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):
        
        input = input.unsqueeze(0) # because only one word, no words sequence 
        embedded = self.dropout(self.embedding(input))
        
        # get weighted sum of encoder_outputs
        '''your code'''
        attention = self.attention(hidden, encoder_outputs)
        weighted_sum = (encoder_outputs*attention).sum(0).unsqueeze(0)
        # concatenate weighted sum and embedded, break through the GRU
        '''your code'''
        output, hidden = self.rnn(torch.cat((embedded,weighted_sum),2), hidden)
        # get predictions
        '''your code'''
        long_ass_vector = torch.cat((embedded,weighted_sum,hidden),2).squeeze(0)
        prediction = self.out(long_ass_vector)
        return prediction, hidden #'''your code'''
        

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.dec_hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        # src = [src sent len, batch size]
        # trg = [trg sent len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        # Again, now batch is the first dimention instead of zero
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        enc_states, hidden = self.encoder(src)
        
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        for t in range(1, max_len):

            output, hidden = self.decoder(input, hidden, enc_states)
            
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            # top1 or ground truth
            input = (trg[t] if teacher_force else top1)
        
        return outputs
