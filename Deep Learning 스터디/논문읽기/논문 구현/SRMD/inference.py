import torch


model = torch.jit.load(r"D:\Model_Inference\save_model\srmd\test.pth")

print(model)