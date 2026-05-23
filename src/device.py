import torch


DGL_MODEL_NAMES = {'GDN', 'MTAD_GAT'}


def get_best_device(requested='auto', model_name=None):
	requested = (requested or 'auto').lower()
	if requested not in {'auto', 'cpu', 'cuda', 'mps'}:
		raise ValueError(f"Unsupported device '{requested}'. Use auto, cpu, cuda, or mps.")

	if requested == 'cuda':
		if not torch.cuda.is_available():
			raise RuntimeError('CUDA was requested, but torch.cuda.is_available() is False.')
		device = torch.device('cuda')
	elif requested == 'mps':
		if not torch.backends.mps.is_available():
			raise RuntimeError('MPS was requested, but torch.backends.mps.is_available() is False.')
		device = torch.device('mps')
	elif requested == 'cpu':
		device = torch.device('cpu')
	elif torch.cuda.is_available():
		device = torch.device('cuda')
	elif torch.backends.mps.is_available():
		device = torch.device('mps')
	else:
		device = torch.device('cpu')

	if model_name in DGL_MODEL_NAMES and device.type == 'mps':
		print(f"DGL model {model_name} does not support MPS; falling back to CPU.")
		device = torch.device('cpu')
	return device


def get_dtype_for_device(device):
	return torch.float32


def move_optimizer_to_device(optimizer, device, dtype=None):
	for state in optimizer.state.values():
		for key, value in state.items():
			if torch.is_tensor(value):
				if dtype is not None and value.is_floating_point():
					state[key] = value.to(device=device, dtype=dtype)
				else:
					state[key] = value.to(device=device)
