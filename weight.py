import torch

# f1 = torch.load('frame_weight.pth')
# print(f1[:100])
# r1 = torch.load('region_weight.pth')
# print(r1[1,1:100])

f = torch.load('frame_weight_single_channelV4_of.pth')
print(f.shape)
print(f[0, 1, :200])
r = torch.load('region_weight_single_channelV4_of.pth')
print(r[0, 1, :200])

