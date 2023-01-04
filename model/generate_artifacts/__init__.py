import torch

# Get PyTorch device to use
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")