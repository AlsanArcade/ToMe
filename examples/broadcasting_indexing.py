import torch

batch_map = torch.arange(2).unsqueeze(1)
token_maps = torch.tensor([[0,1],[1,0]])
t_orig = torch.tensor(
        [[[2, 1],
         [1, 4]],

        [[1, 1],
         [4, 3]]])
print(t_orig)
t_reference = t_orig[:,token_maps[1],:]
print(t_reference)
t_new = t_orig[batch_map,token_maps,:]
print(t_new)