import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchtext.datasets import Multi30k
from collections import Counter
import spacy
import random

# Set seed
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# Load spaCy tokenizers
spacy_de = spacy.load("de_core_news_sm")
spacy_en = spacy.load("en_core_web_sm")

def tokenize_de(text): return [tok.text.lower() for tok in spacy_de.tokenizer(text)]
def tokenize_en(text): return [tok.text.lower() for tok in spacy_en.tokenizer(text)]

# Special tokens
SOS = "<sos>"
EOS = "<eos>"
PAD = "<pad>"
UNK = "<unk>"

# Build vocabulary
def build_vocab(sentences, tokenizer, min_freq=2):
    counter = Counter()
    for sent in sentences:
        tokens = tokenizer(sent)
        counter.update(tokens)
    vocab = {PAD: 0, SOS: 1, EOS: 2, UNK: 3}
    for word, freq in counter.items():
        if freq >= min_freq and word not in vocab:
            vocab[word] = len(vocab)
    return vocab

# Numericalization
def numericalize(tokens, vocab):
    return [vocab.get(tok, vocab[UNK]) for tok in tokens]

# Custom dataset
class TranslationDataset(Dataset):
    def __init__(self, split, src_vocab=None, trg_vocab=None):
        raw_data = list(Multi30k(split=split, language_pair=('de', 'en')))
        self.src_sentences, self.trg_sentences = zip(*raw_data)
        self.src_vocab = src_vocab or build_vocab(self.src_sentences, tokenize_de)
        self.trg_vocab = trg_vocab or build_vocab(self.trg_sentences, tokenize_en)
        self.inv_trg_vocab = {v: k for k, v in self.trg_vocab.items()}

    def __len__(self): return len(self.src_sentences)

    def __getitem__(self, idx):
        src = [SOS] + tokenize_de(self.src_sentences[idx]) + [EOS]
        trg = [SOS] + tokenize_en(self.trg_sentences[idx]) + [EOS]
        return numericalize(src, self.src_vocab), numericalize(trg, self.trg_vocab)

# Collate function for padding
def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    src_lens = [len(x) for x in src_batch]
    trg_lens = [len(x) for x in trg_batch]
    max_src = max(src_lens)
    max_trg = max(trg_lens)

    def pad(seq, max_len): return seq + [0] * (max_len - len(seq))

    src_padded = torch.tensor([pad(seq, max_src) for seq in src_batch], dtype=torch.long).T
    trg_padded = torch.tensor([pad(seq, max_trg) for seq in trg_batch], dtype=torch.long).T
    return src_padded, trg_padded

# Load dataset
train_data = TranslationDataset(split='train')
train_loader = DataLoader(train_data, batch_size=128, shuffle=True, collate_fn=collate_fn)

SRC_VOCAB_SIZE = len(train_data.src_vocab)
TRG_VOCAB_SIZE = len(train_data.trg_vocab)
PAD_IDX = train_data.trg_vocab[PAD]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=0)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell

# Decoder
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=0)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)  # [1, batch]
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        return self.fc_out(output.squeeze(0)), hidden, cell

# Seq2Seq model
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        trg_len, batch_size = trg.shape
        trg_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        hidden, cell = self.encoder(src)
        input = trg[0, :]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            top1 = output.argmax(1)
            input = trg[t] if random.random() < teacher_forcing_ratio else top1

        return outputs

# Instantiate model
enc = Encoder(SRC_VOCAB_SIZE, 256, 512, 2, 0.5)
dec = Decoder(TRG_VOCAB_SIZE, 256, 512, 2, 0.5)
model = Seq2Seq(enc, dec, device).to(device)

# Optimizer and loss
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# Training loop
def train(model, iterator):
    model.train()
    epoch_loss = 0
    for src, trg in iterator:
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg)
        output = output[1:].reshape(-1, output.shape[-1])
        trg = trg[1:].reshape(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

# Run one epoch
print("Training started...")
for epoch in range(12):
    loss = train(model, train_loader)
    print(f"Epoch {epoch} Loss: {loss:.3f}")


def translate_sentence(sentence, model, src_vocab, trg_vocab, tokenizer, max_len=50):
    model.eval()
    
    # Tokenize and numericalize
    tokens = [SOS] + tokenizer(sentence.lower()) + [EOS]
    src_indexes = [src_vocab.get(tok, src_vocab[UNK]) for tok in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)  # [src_len, 1]

    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor)

    trg_indexes = [trg_vocab[SOS]]
    for _ in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
        pred_token = output.argmax(1).item()
        if pred_token == trg_vocab[EOS]:
            break
        trg_indexes.append(pred_token)

    # Reverse trg vocab
    inv_trg_vocab = {v: k for k, v in trg_vocab.items()}
    translated_tokens = [inv_trg_vocab[i] for i in trg_indexes[1:]]
    return " ".join(translated_tokens)

example = "Ein kleines Mädchen"
translation = translate_sentence(
    example,
    model,
    train_data.src_vocab,
    train_data.trg_vocab,
    tokenize_de
)
print(f"\nGerman: {example}")
print(f"English (predicted): {translation}")
