#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2024/2025
#############################################################################
###
### Домашно задание 3
###
#############################################################################

import numpy as np
import torch
import torch.nn.functional as F
from lev_dist import rhyme_dist_between_suffixes
from accentor import StressTransformer

def generate_line(model, start_char_idx, h_state=None, c_state=None, max_len=100, temperature=0.4):
	# генерира един докато не срещне символ за край или нов ред.
	model.eval()
	device = next(model.parameters()).device
	
	ind2char = {v: k for k, v in model.char2ind.items()}
	
	curr_char_idx = torch.tensor([[start_char_idx]], dtype=torch.long, device=device)
	
	if h_state is None:
		h_state = torch.zeros(model.lstm_layers, 1, model.hidden_size, device=device)
		c_state = torch.zeros(model.lstm_layers, 1, model.hidden_size, device=device)
	
	generated_indices = []
	
	with torch.no_grad():
		for _ in range(max_len):
			emb = model.embed_char(curr_char_idx)
			
			output, (h_state, c_state) = model.lstm(emb, (h_state, c_state))
			
			output = model.dropout(output)
			logits = model.projection(output.squeeze(0))
			
			probs = F.softmax(logits / temperature, dim=-1)
			next_char_idx = torch.multinomial(probs, 1)
			
			idx_val = next_char_idx.item()
			
			if idx_val == model.endToken or idx_val == model.char2ind['\n']:
				break
				
			generated_indices.append(idx_val)
			curr_char_idx = next_char_idx

	generated_str = "".join([ind2char.get(idx, "") for idx in generated_indices])
	
	return generated_str, h_state, c_state


def get_post_stress_suffix(text: str, stress_model: StressTransformer) -> str:
	if not text:
		return ""
	
	words = text.strip().split()
	if not words:
		return ""
	last_word = words[-1]

	stress_pattern = stress_model.inference(last_word)
	stress_idx = stress_pattern.find('1')

	if stress_idx != -1:
		return last_word[stress_idx + 1:]
	else:
		return last_word


def generate_aabb_poem(model, stress_model, start_char, num_stanzas=4):
	# Генерира стих с AABB рима. 
	# Всеки нечетен стих се генерира свободно, а всеки четен се генерира К=32 пъти
	# от тези К кандидати избираме този с най-малко Левенщайн разстояние между 
	# сифукса на последната дума на предният ред след ударената гласна и суфикса на аналогичната 
	# на нея дума на следващият ред.
	K_SAMPLES = 32
	start_token_idx = model.char2ind[start_char]
	newline_idx = model.char2ind['\n']

	device = next(model.parameters()).device
	h = torch.zeros(model.lstm_layers, 1, model.hidden_size, device=device)
	c = torch.zeros(model.lstm_layers, 1, model.hidden_size, device=device)

	final_poem = []

	current_start_idx = start_token_idx
	first_line_prefix = start_char 

	for _ in range(num_stanzas * 2):
		line_1_body, h_1, c_1 = generate_line(model, current_start_idx, h, c)
		
		if current_start_idx == start_token_idx:
			line_1_full = first_line_prefix + line_1_body
		else:
			line_1_full = line_1_body
			
		final_poem.append(line_1_full)
		target_suffix = get_post_stress_suffix(line_1_full, stress_model)
		next_input_idx = newline_idx
		candidates = []

		for _ in range(K_SAMPLES):
			cand_str, cand_h, cand_c = generate_line(model, next_input_idx, h_1, c_1)
			cand_suffix = get_post_stress_suffix(cand_str, stress_model)
			
			candidates.append({
				'text': cand_str,
				'h': cand_h,
				'c': cand_c,
				'suffix': cand_suffix
			})
		
		best_candidate = min(candidates, key=lambda x: rhyme_dist_between_suffixes(target_suffix, x['suffix']))
		
		line_2 = best_candidate['text']
		final_poem.append(line_2)

		h, c = best_candidate['h'], best_candidate['c']
		
		current_start_idx = newline_idx

	return "\n".join(final_poem)[1:]


def generate_poem(model, start_char, limit=1000, temperature=0.4):
	# Генерира поема по стандартният начин, докато не стигне лимита или не срещне символ за край.
	model.eval()
	device = next(model.parameters()).device
	start_char_idx = model.char2ind[start_char]
	ind2char = {v: k for k, v in model.char2ind.items()}

	h_state = torch.zeros(model.lstm_layers, 1, model.hidden_size, device=device)
	c_state = torch.zeros(model.lstm_layers, 1, model.hidden_size, device=device)

	curr_char_idx = torch.tensor([[start_char_idx]], dtype=torch.long, device=device)

	generated_indices = []

	with torch.no_grad():
		for _ in range(limit):
			emb = model.embed_char(curr_char_idx)
			
			output, (h_state, c_state) = model.lstm(emb, (h_state, c_state))
			
			output = model.dropout(output)
			logits = model.projection(output.squeeze(0))
			
			probs = F.softmax(logits / temperature, dim=-1)
			next_char_idx = torch.multinomial(probs, 1)
			
			idx_val = next_char_idx.item()
			
			generated_indices.append(idx_val)
			curr_char_idx = next_char_idx
			
			if idx_val == model.endToken:
				break

	generated_str = "".join([ind2char.get(idx, "") for idx in generated_indices])

	return generated_str[:-1]