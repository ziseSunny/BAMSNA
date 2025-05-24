import torch
import torch.nn.functional as F
	
def feature_reconstruct_loss(embd, x, recon_model):
	recon_x = recon_model(embd)
	return torch.norm(recon_x - x, dim=1, p=2).mean()