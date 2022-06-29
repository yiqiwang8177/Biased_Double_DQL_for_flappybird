"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch.nn as nn
import torch

class DeepQNetwork(nn.Module):
    def __init__(self, actions = 2):
        super(DeepQNetwork, self).__init__()
        self.actions = actions
#         self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=2) # 
#         self.relu1 = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
#         self.relu2 = nn.ReLU(inplace=True)

#         self.fc1 = nn.Linear(6400, 256) # hard coded
#         self.relu3 = nn.ReLU(inplace=True)
#         self.fc2 = nn.Linear(256, self.actions)
        self.conv1 = nn.Sequential(
          nn.Conv2d(4, 32, kernel_size=8, stride=4),
          nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
          nn.Conv2d(32, 64, kernel_size=4, stride=2),
          nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
          nn.Conv2d(64, 64, kernel_size=3, stride=1),
          nn.ReLU(inplace=True)
        )
        self.fc1 = nn.Sequential(
          nn.Linear(in_features=3136, out_features=512, bias=True),
          nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Linear(in_features=512, out_features=2, bias=True)
   
    def get_q_value(self, o):
        """Get Q value estimation w.r.t. current observation `o`
           o -- current observation
        """


        out = self.conv1(o)

        out = self.conv2(out)

        out = self.conv3(out)

        out = out.view(out.size()[0], -1)

        out = self.fc1(out)
        out = self.fc2(out)
        return out

    def _create_weights(self):
        raise NotImplementedError

    def forward(self, input):
        return self.get_q_value(input)
    
#     def get_optim_action(self):
#         """Get optimal action based on current state
#         """
#         state = self.current_state
#         state_var = Variable(torch.from_numpy(state), volatile=True).unsqueeze(0)
#                 if self.use_cuda:
#                     state_var = state_var.cuda()
#         q_value = self.forward(state_var)
#         _, action_index = torch.max(q_value, dim=1)
#         action_index = action_index.data[0][0]
#         action = np.zeros(self.actions, dtype=np.float32)
#         action[action_index] = 1
#         return action
