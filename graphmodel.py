import torch
import torch.nn.functional as F
from directednet import DirectedUnweighted

class LGCNDirected(torch.nn.Module):
	def __init__(self, input_size, output_size, hidden_size=512, K=2):
		super(LGCNDirected, self).__init__()
		self.conv1 = DirectedUnweighted(K=K)
		self.linear = torch.nn.Linear(input_size * 2 * (K + 1), output_size) #K+1：K为conv层数，1为原始的X
	def forward(self, feature, edge_index):
		x = self.conv1(feature, edge_index)
		x = self.linear(x)
		return x


class WDiscriminator(torch.nn.Module):
	def __init__(self, hidden_size, hidden_size2=512):
		super(WDiscriminator, self).__init__()
		self.hidden = torch.nn.Linear(hidden_size, hidden_size2)
		self.hidden2 = torch.nn.Linear(hidden_size2, hidden_size2)
		self.output = torch.nn.Linear(hidden_size2, 1)
	def forward(self, input_embd):
		return self.output(F.leaky_relu(self.hidden2(F.leaky_relu(self.hidden(input_embd), 0.2, inplace=True)), 0.2, inplace=True))

class transformation(torch.nn.Module):
	def __init__(self, hidden_size=512, hidden_size2=512):
		super(transformation, self).__init__()
		self.trans = torch.nn.Parameter(torch.eye(hidden_size))
	def forward(self, input_embd):
		return input_embd.mm(self.trans)
	
class notrans(torch.nn.Module):
	def __init__(self):
		super(notrans, self).__init__()
	def forward(self, input_embd):
		return input_embd

class ReconDNN(torch.nn.Module):
	def __init__(self, hidden_size, feature_size, hidden_size2=512):
		super(ReconDNN, self).__init__()
		self.hidden = torch.nn.Linear(hidden_size, hidden_size2)
		self.output = torch.nn.Linear(hidden_size2, feature_size)
	def forward(self, input_embd):
		return self.output(F.relu(self.hidden(input_embd)))