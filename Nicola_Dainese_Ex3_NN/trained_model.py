import torch
import argparse
import numpy as np
import preprocessing as prep
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class Network(nn.Module):
    
    def __init__(self, vocab_size, emb_dim, hidden_units, layers_num, dropout_prob=0, linear_size=512):
        super().__init__()
        # Define recurrent layer
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        
        self.rnn = nn.LSTM(input_size=emb_dim, 
                           hidden_size=hidden_units,
                           num_layers=layers_num,
                           dropout=dropout_prob,
                           batch_first=True)
        
        self.l1 = nn.Linear(hidden_units,linear_size)
        self.out = nn.Linear(linear_size,vocab_size) # leave out the '<PAD>' label 
        
    def forward(self, x, seq_lengths, state=None):

        # Embedding of x
        x = self.embedding(x) 
        
        # packing for efficient processing
        packed_input = pack_padded_sequence(x, seq_lengths, batch_first=True)
        
        # propagate through the LSTM
        packed_output, state = self.rnn(packed_input, state)
        
        # unpack for linear layers processing
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        #print("Unpacked output: ", output.size(), '\n')
        
        # Linear layer
        output = F.leaky_relu(self.l1(output))

        # Linear layer with log_softmax (useful for custom cross-entropy)
        output = F.log_softmax(self.out(output), dim=2)

        return output, state
    
def generate_sentence(seed, trained_net, word2index, index2word, len_generated_seq = 10, debug=False, T=1):
    
    # preprocess seed
    import re
    special_chars = ['!','?','&','(',')','*','-','_',':',';','"','\'','1','2','3','4','5','6','7','8','9','0']
    for x in special_chars:
        seed = seed.replace(x,' ')
    full_text = seed.lower()
    full_text = full_text.replace('mr.','mr')
    full_text = full_text.replace('mrs.','mrs')
    full_text = re.sub('à', 'a', full_text)
    full_text = re.sub('ê', 'e', full_text)
    full_text = re.sub(r'[.]',' .\n', full_text)
    full_text = full_text.replace(',',' ,')
    full_text = full_text.replace('  ',' ')
    
    num_sentence = prep.numerical_encoder(full_text, word2index)
    if debug:
        for i,num_w in enumerate(num_sentence):
            print(i,num_w)
        
    enc_sentence = torch.LongTensor(num_sentence).view(1,-1)
    context = enc_sentence[:,:-1]
    length_context = torch.LongTensor(np.array([len(num_sentence)-1])).view(1)
    last_word = enc_sentence[:,-1].view(1,1)
    length_last = torch.LongTensor([1]).view(1)
    
    if debug:
        print("enc_sentence : ", enc_sentence.size())
        print("context : ", context.size(), context)
        print("length_context: ", length_context.size(), length_context)
        print("last_word: ", last_word.size(), last_word)
        
    with torch.no_grad():
        net.eval()
        _, hidden_context = net(context, length_context)

        gen_words = []
        for i in range(len_generated_seq):
            last_word_ohe, hidden_context = net(last_word, length_last, state=hidden_context)
            prob_last_word = np.exp(last_word_ohe.numpy().flatten()/T)
            prob_last_word = prob_last_word/ prob_last_word.sum()
            
            if debug:
                print("prob_last_word (shape): ", prob_last_word.shape)
                print("Sum of probabilities: ", prob_last_word.sum())
                print("'<PAD>' probability: ", prob_last_word[0])
            last_word_np = np.random.choice(np.arange(len(prob_last_word)), p=prob_last_word)
            gen_words.append(last_word_np)
            last_word = torch.LongTensor([last_word_np]).view(1,1)
            if debug:
                print("Last word: ", last_word_np)
    
    gen_words = np.array(gen_words).flatten()
    decoded_sentence = prep.numerical_decoder(gen_words, index2word)
    output_string = ' '.join(decoded_sentence)
    print("Seed: ", seed, '\n')
    print("Generated sentence: ", output_string)
    print("\nAll toghether: ", seed, output_string)
    
##############################
##############################
## PARAMETERS
##############################
parser = argparse.ArgumentParser(description='Generate a chapter starting from a given text')

parser.add_argument('--seed', type=str, default='there was', help='Initial text of the chapter')
parser.add_argument('--model_dir',   type=str, default='austen_model', help='Network model directory')
parser.add_argument('--length', type=int, default=10, help='Number of words to be generated')
parser.add_argument('--T', type=float, default=1, help='Thermal noise factor for choosing the nest word from the probabilities (T > 0, the smaller the less variability there is)')

if __name__ == '__main__':
    
    ### Parse input arguments
    args = parser.parse_args()
    
    # load word dictionaries
    dataset = np.load(args.model_dir+"/dataset.npy", allow_pickle=True).item()
    word2index = dataset['word2index']
    index2word = dataset['index2word']
    
    params = dict(vocab_size=len(word2index), emb_dim=100, hidden_units=128, layers_num=2, dropout_prob=0.2)
    net = Network(**params)
    net.load_state_dict(torch.load(args.model_dir+'/final_params.pth'))
    generate_sentence(args.seed, net, word2index, index2word, len_generated_seq=args.length, T=args.T, debug=False)