from typing import (
	Any,
	Callable,
	Dict,
	List,
	Optional,
	Sequence,
	Tuple,
	Type,
	Union,
	no_type_check,
)

import numpy as np
import torch
import torch.nn as nn

from unit.GRU import Encoder, Attention, Decoder

ModuleType = Type[nn.Module]

class Transformer_GRU(nn.Module):
	"""
	Transformer structure constructing gym-like training
	"""
	def __init__(
		self,
		state_shape: Union[int, Sequence[int]],
		action_shape: Union[int, Sequence[int]],
		device: Union[str, int, torch.device] = "cpu",
		hidden_layer_size: int = 256,
		embedding_dim: int = 256,
	) -> None:
		super().__init__()
		self.device = device
		self.state_shape = state_shape
		self.hidden = torch.tensor([])
		
		self.fc1 = nn.Linear(int(np.prod(state_shape)), hidden_layer_size)
		self.fc2 = nn.Linear(hidden_layer_size, int(np.prod(action_shape)))

		self.encoder = Encoder(int(np.prod(state_shape)), embedding_dim, hidden_layer_size)
		self.attention = Attention()
		self.decoder = Decoder(int(np.prod(action_shape)), embedding_dim, hidden_layer_size)

	def forward(
		self,
		obs: Union[np.ndarray, torch.Tensor],
		state: Any = None,
		info: Dict[str, Any] = {},
	) -> Tuple[torch.Tensor, Any]:
		"""Mapping: obs -> flatten -> logits.
		In the evaluation mode, `obs` should be with shape ``[bsz, dim]``; in the
		training mode, `obs` should be with shape ``[bsz, len, dim]``. See the code
		and comment for more detail.
		"""
		obs = torch.as_tensor(
			obs,
			device=self.device,
			dtype=torch.float32,
		)
		# obs [bsz, len, dim] (training) or [bsz, dim] (evaluation)
		# In short, the tensor's shape in training phase is longer than which
		# in evaluation phase.

		obs = torch.reshape(obs, (-1, int(np.prod(self.state_shape))))

		obs_fc1 = self.fc1(obs)
		obs_fc2 = self.fc2(obs_fc1)
		
		if len(self.hidden) == 0:
			enc_hidden = self.encoder.initialize_hidden_state(int(obs.shape[0]))
			enc_output, dec_hidden = self.encoder(obs.long(), enc_hidden)
		elif self.hidden.shape[1] != obs.shape[0]:
			self.hidden = self.hidden.max(1).values.unsqueeze(1)
			self.hidden = self.hidden.repeat(1, obs.shape[0], 1)
			enc_output, dec_hidden = self.encoder(obs.long(), self.hidden)
		else:
			enc_output, dec_hidden = self.encoder(obs.long(), self.hidden)

		obs, hn = self.decoder(obs_fc2.long(), dec_hidden, enc_output)

		self.hidden = hn.clone().detach()

		return obs, state