import torch
import torch.nn as nn
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load vocabularies
with open("vocab_src.pkl", "rb") as f:
    vocab_src = pickle.load(f)
with open("vocab_tgt.pkl", "rb") as f:
    vocab_tgt = pickle.load(f)

idx2word_tgt = {idx: word for word, idx in vocab_tgt.items()}

pad_idx_src = vocab_src["<pad>"]
pad_idx_tgt = vocab_tgt["<pad>"]

def tokenize(text):
    return text.lower().split()

def numericalize(text, vocab):
    return [vocab.get(tok, vocab["<unk>"]) for tok in tokenize(text)]

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx_src)
        self.rnn = nn.GRU(emb_dim, hid_dim)
    def forward(self, src):
        embedded = self.embedding(src)
        _, hidden = self.rnn(embedded)
        return hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=pad_idx_tgt)
        self.rnn = nn.GRU(emb_dim, hid_dim)
        self.fc_out = nn.Linear(hid_dim, output_dim)
    def forward(self, input, hidden):
        input = input.unsqueeze(0)
        embedded = self.embedding(input)
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

INPUT_DIM = len(vocab_src)
OUTPUT_DIM = len(vocab_tgt)
EMB_DIM = 64
HID_DIM = 128

encoder = Encoder(INPUT_DIM, EMB_DIM, HID_DIM).to(device)
decoder = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM).to(device)
model = Seq2Seq(encoder, decoder).to(device)

model.load_state_dict(torch.load("translation_model.pth", map_location=device))
model.eval()
