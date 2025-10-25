import torch
print(torch.version.cuda)      # Should print the CUDA version PyTorch is using
print(torch.cuda.is_available())  # Should print True if GPU is usable
