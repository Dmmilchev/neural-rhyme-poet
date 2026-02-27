import torch

def softmin(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, gamma: float) -> torch.Tensor:
    x = torch.stack([a, b, c], dim=0)
    return -gamma * torch.logsumexp(-x / gamma, dim=0)

def soft_edit_distance(x1: torch.Tensor, x2: torch.Tensor, ins_cost: float, gamma: float) -> torch.Tensor:
	# алгоритъмът на Вагнер-Фишър с единствената разлика, че вместо минимум на ред 26, 
	# ползваме изгладен минимум, за да имаме възможността да диференцираме.
	# x1.shape = (seq_len1, vocab_size)
	# x2.shape = (seq_len2, vocab_size)	
	seq_len1, vocab_size = x1.shape
	seq_len2, _ = x2.shape

	dist = torch.cdist(x1, x2, p=2)
	
	dp = torch.empty((seq_len1 + 1, seq_len2 + 1), dtype=torch.float32, device=x1.device)
	dp[0,0] = 0.0
	for i in range(1, seq_len1 + 1):
		dp[i,0] = dp[i-1,0] + ins_cost
	for j in range(1, seq_len2 + 1):
		dp[0,j] = dp[0,j-1] + ins_cost

	for i in range(1, seq_len1 + 1):
		for j in range(1, seq_len2 + 1):
			dp[i,j] = softmin(dp[i-1,j] + ins_cost, dp[i,j-1] + ins_cost, dp[i-1,j-1] + dist[i-1,j-1], gamma)

	return dp[seq_len1, seq_len2]