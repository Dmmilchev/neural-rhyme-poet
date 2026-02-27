import torch
import torch.nn.functional as F
from sed import soft_edit_distance
from accentor import StressTransformer
from parameters import INS_COST, GAMMA


class LSTMLanguageModelPack(torch.nn.Module):
    def preparePaddedBatch(self, source):
        device = next(self.parameters()).device
        m = max(len(s) for s in source)
        sents = [[self.char2ind.get(w,self.unkTokenIdx) for w in s] for s in source]
        sents_padded = [ s+(m-len(s))*[self.padTokenIdxChar2ind] for s in sents]
        return torch.t(torch.tensor(sents_padded, dtype=torch.long, device=device))
    
    def save(self,fileName):
        torch.save(self.state_dict(), fileName)
    
    def load(self,fileName,device):
        self.load_state_dict(torch.load(fileName,device))

    def __init__(self, embed_size, hidden_size, char2ind, unkToken, padToken, endToken, lstm_layers, dropout):
        super(LSTMLanguageModelPack, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.char2ind = char2ind
        self.ind2char = {v: k for k, v in char2ind.items()}
        self.unkTokenIdx = char2ind[unkToken]
        self.padTokenIdxChar2ind = char2ind[padToken]
        self.endToken = char2ind[endToken]
        self.lstm_layers = lstm_layers
        self.lstm = torch.nn.LSTM(input_size=embed_size, 
                                  hidden_size=hidden_size, 
                                  num_layers=lstm_layers)
        self.embed_char = torch.nn.Embedding(len(char2ind), embed_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.projection = torch.nn.Linear(hidden_size,len(char2ind))
        

    def forward(self, source, stress_model):
        padded_sents = self.preparePaddedBatch(source)
        # padded_sents.shape = (max_seq_len, batch_size)
        sents_embedded = self.embed_char(padded_sents[:-1])
        # sents_embedded.shape = (max_seq_len - 1, batch_size, char_emb_size)

        device = padded_sents.device
        batch_size = padded_sents.size(1)

        h_0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_size, device=device)
        c_0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_size, device=device)
        
        source_lengths = [len(s)-1 for s in source]
        outputPacked, (h_n, c_n) = self.lstm(
            torch.nn.utils.rnn.pack_padded_sequence(sents_embedded,
                                                    source_lengths,
                                                    enforce_sorted=False), 
             (h_0, c_0))
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(outputPacked)
        # output.shape = (max_seq_len - 1, batch_size, hidden_size)
        output = self.dropout(output)
        Z = self.projection(output)
        # Z.shape = ((max_seq_len - 1),  batch_size, len(char2ind))
        Z_flattened = Z.flatten(0, 1)
        Y_bar = padded_sents[1:].flatten(0, 1)
        H = torch.nn.functional.cross_entropy(Z_flattened, Y_bar, ignore_index=self.padTokenIdxChar2ind)

		# до тук свършва стандартното обуние на езиковият модел.
		# от тук започва пресмятането на грешката за рима.
        newline_id = self.char2ind['\n']
        batch_size = padded_sents.size(1)
        vocab_size = len(self.char2ind)
        device = next(self.parameters()).device
        sed_loss = torch.tensor(0.0, device=device)
        total_pairs = 0
        punctuation_set = set(['.', ',', '!', '?', ':', ';', '-', '…'])

        for b in range(batch_size):
            nl_indices = (padded_sents[:, b] == newline_id).nonzero(as_tuple=False).flatten().tolist()

            if len(nl_indices) < 2:
                continue

            lines_boundaries = []
            current_start = 0
            for nl in nl_indices:
                lines_boundaries.append((current_start, nl))
                current_start = nl + 1

            for i in range(1, len(lines_boundaries), 2):
                prev_start, prev_end = lines_boundaries[i-1]

                prev_end_np = prev_end
                while prev_end_np > prev_start:
                    char_idx = padded_sents[prev_end_np - 1, b].item()
                    char_val = self.ind2char[char_idx]
                    
                    if char_val in punctuation_set:
                        prev_end_np -= 1
                    else:
                        break
				
                if prev_end_np <= prev_start:
                    continue

                p_slice_start = -1
                prev_line_chars = padded_sents[prev_start:prev_end_np, b]
                
                for idx in range(len(prev_line_chars) - 1, -1, -1):
                    charid = prev_line_chars[idx].item()
                    if charid == self.char2ind[' ']:
                        p_slice_start = prev_start + idx
                        break
                p_slice_start = max(prev_start, p_slice_start)
                
                if p_slice_start >= prev_end_np:
                    continue

                gt_indices = padded_sents[p_slice_start:prev_end_np, b]
				# до тук сме взели последната дума от ред i-1, без пунктуацията, накрая.
				# в противен случай, моделът се научава да повтаря пунктуацията в краят на реда.
                gt_text = ''.join([self.ind2char[i] for i in gt_indices.tolist()])
                gt_stress = stress_model.inference(gt_text).find('1')
                if gt_stress == -1:
                    continue
                gt_onehot = F.one_hot(gt_indices[gt_stress:], num_classes=vocab_size).float().to(device)
                curr_start, curr_end = lines_boundaries[i]
                
                suffix_len = len(gt_indices) - gt_stress
                c_slice_start = max(curr_start, curr_end - suffix_len)
                # до тук сме взели наставката на думата след ударената гласна, включително самата ударна гласна.

                if c_slice_start >= curr_end:
                    continue

                z_slice = Z[c_slice_start-1 : curr_end-1, b]
                
                pred_probs = F.softmax(z_slice, dim=-1)

				# сравняваме наставката на думата на предният ред със съответният следващ ред, който моделът генерира.
                loss_val = soft_edit_distance(pred_probs, gt_onehot, INS_COST, GAMMA)
                
                sed_loss += loss_val
                total_pairs += 1
            
        lambda_sed = 0.25
        
        if total_pairs > 0:
            sed_loss = sed_loss / total_pairs
            
        return H + (lambda_sed * sed_loss), H, sed_loss

    def get_H(self, source):
        # Само първата част от forward, без грешката за рима.

        padded_sents = self.preparePaddedBatch(source)
        # padded_sents.shape = (max_seq_len, batch_size)
        sents_embedded = self.embed_char(padded_sents[:-1])
        # sents_embedded.shape = (max_seq_len - 1, batch_size, char_emb_size)

        device = padded_sents.device
        batch_size = padded_sents.size(1)

        h_0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_size, device=device)
        c_0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_size, device=device)

        source_lengths = [len(s)-1 for s in source]
        outputPacked, (h_n, c_n) = self.lstm(
            torch.nn.utils.rnn.pack_padded_sequence(sents_embedded,
                                                    source_lengths,
                                                    enforce_sorted=False), 
                (h_0, c_0))
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(outputPacked)
        # output.shape = (max_seq_len - 1, batch_size, hidden_size)
        output = self.dropout(output)
        Z = self.projection(output)
        # Z.shape = ((max_seq_len - 1),  batch_size, len(char2ind))
        Z_flattened = Z.flatten(0, 1)
        Y_bar = padded_sents[1:].flatten(0, 1)
        H = torch.nn.functional.cross_entropy(Z_flattened, Y_bar, ignore_index=self.padTokenIdxChar2ind)
        
        return H