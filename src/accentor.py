import torch
import torch.nn as nn
import math

# Global constants as defined in your prompt
startChar = '{'
endChar = '}'
unkChar = '@'
padChar = '|'
alphabet = '}{@|АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЬЮЯабвгдежзийклмнопрстуфхцчшщъьюя-`'
char2ind = {k:i for i, k in enumerate(alphabet)}

class PositionalEncoding(nn.Module):
	# Стандартното позиционно влагане от статията attention is all you need.
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(1)) 

    def forward(self, x):
        return x + self.pe[:x.size(0)]


class StressTransformer(nn.Module):
	# Стандартен трансформър блок само с декодер. 
    def __init__(self, vocab_map=None, d_model=64, nhead=4, num_layers=2):
        super(StressTransformer, self).__init__()
        
        self.char2ind = {v: k for k, v in enumerate(alphabet)}
        self.padTokenIdx = self.char2ind.get(padChar, 0)
        self.unkTokenIdx = self.char2ind.get(unkChar, 1)
        
        self.embedding = nn.Embedding(len(self.char2ind), d_model, padding_idx=self.padTokenIdx)
        
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.fc = nn.Linear(d_model, 1)

    def preparePaddedBatch(self, source):
        device = next(self.parameters()).device
        m = max(len(w) for (w, s) in source)
        sents = [[self.char2ind.get(w, self.unkTokenIdx) for w in word] for (word, stress) in source]
        sents_padded = [s + (m - len(s)) * [self.padTokenIdx] for s in sents]
        labels = [[int(c) for c in stress] for (word, stress) in source]
        labels_padded = [l + (m - len(l)) * [-1.0] for l in labels]
        inputs = torch.t(torch.tensor(sents_padded, dtype=torch.long, device=device))
        targets = torch.t(torch.tensor(labels_padded, dtype=torch.float, device=device))
        
        return inputs, targets

    def forward(self, src):
        src_key_padding_mask = (src == self.padTokenIdx).transpose(0, 1)
        x = self.embedding(src)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        prediction_logits = self.fc(output)
        return prediction_logits.squeeze(-1)

    def inference(self, word: str) -> str:
		# Генерира стринга от 0 и 1, който показва къде е ударената гласна.
        if word == '': return ''
        for c in word:
            if c not in self.char2ind.keys():
                None
        ids = [self.char2ind.get(c, self.unkTokenIdx) for c in word]
        device = next(self.parameters()).device
        input_tensor = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(1)
        with torch.no_grad():
            logits = self.forward(input_tensor)
            probs = torch.sigmoid(logits).squeeze(1)
            stress_string = ''.join(['1' if p > 0.5 else '0' for p in probs.cpu().numpy()])
            return stress_string

