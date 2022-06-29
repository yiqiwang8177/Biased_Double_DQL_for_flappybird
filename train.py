"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import argparse
import os
import shutil
from random import random, randint, sample

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from src.deep_q_network import DeepQNetwork
from src.flappy_bird import FlappyBird
from src.utils import pre_processing

from collections import deque
import random
import copy


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Flappy Bird""")
    parser.add_argument("--image_size", type=int, default=84, help="The common width and height for all images")
    parser.add_argument("--batch_size", type=int, default=32, help="The number of images per batch")
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adam"], default="adam")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=0.1)
    parser.add_argument("--final_epsilon", type=float, default=1e-4)
    parser.add_argument("--num_iters", type=int, default=2000000)
    parser.add_argument("--replay_memory_size", type=int, default=50000,
                        help="Number of epoches between testing phases")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")

    args = parser.parse_args()
    return args


def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    model = DeepQNetwork()
    model2 = copy.deepcopy(model)
    for param in model2.parameters():
          param.requires_grad = False
            
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
    criterion = nn.MSELoss()
    game_state = FlappyBird()
    image, reward, terminal = game_state.next_frame(0)
    image = pre_processing(image[:game_state.screen_width, :int(game_state.base_y)], opt.image_size, opt.image_size)
    image = torch.from_numpy(image)
    if torch.cuda.is_available():
        model.cuda()
        image = image.cuda()
    state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]

    replay_memory = deque( maxlen = opt.replay_memory_size)
    iter = 0
    epsilon_gap = opt.initial_epsilon - opt.final_epsilon
    actions = [0,1]
    epsilon = opt.initial_epsilon 
    while iter < opt.num_iters:

        # 你可以在此：
        # 1。输出你模型的预测
        # 2。选择你的EPSILON， 输出变量名: epsilon
        # 3。使用 EPSILON greedy ，输出
        #TODO

        q_value = model(state)
        _, action_index = torch.max(q_value, dim=1)
        
#         epsilon = opt.initial_epsilon - iter* ( epsilon_gap / opt.num_iters) 
        if epsilon > opt.final_epsilon:
            epsilon -= 0.000005
        else:
            epsilon = opt.final_epsilon
        prob = random.uniform(0, 1)

        if prob >= epsilon:
            action = actions[action_index]
        else:
#             action = random.choice([0,1]) # most of time be lazy
            action = random.choice([0,0,0,0,0,0,0,0,0,0,1,0,0,0]) # most of time be lazy
#             if iter > 20000:
#                 action = random.choice(actions) # now, no RL implemented, random action

        if iter % 20 == 0:
            print(f"Iter:{iter}\tq_value:{q_value}\tepsilon:{epsilon}\taction_index:{action_index.item()}\taction:{action}\t")
        
        next_image, reward, terminal = game_state.next_frame(action)
        next_image = pre_processing(next_image[:game_state.screen_width, :int(game_state.base_y)], opt.image_size,
                                    opt.image_size)
        next_image = torch.from_numpy(next_image) # shape is: 1, 84, 84
        
        if torch.cuda.is_available():
            next_image = next_image.cuda()
        next_state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]

        # 把 state, action, reward, next_state, terminal 放在一个LIST 中，并加入到replay_memory 中
        # 如果 replay_memory 满了（> opt.replay_memory_size），请做相应处理（删除）
        # 从replay buffer 中sample 出BATCH 进行训练
        #TODO
        replay_memory.append([state, action, reward, next_state, terminal]) # replay_mem is a deque()

        batch = sample(replay_memory, min(len(replay_memory), opt.batch_size) )

        # 此代处对sammple 出来的结果进行拆分
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = zip(*batch)

        state_batch = torch.cat(tuple(state for state in state_batch))
        action_batch = torch.from_numpy(
            np.array([[1, 0] if action == 0 else [0, 1] for action in action_batch], dtype=np.float32))
        action_batch2 = torch.from_numpy( # use for double-dqn
            np.array([[True, False]
                      if torch.max(model(next_state),dim=1)[1] == 0 else [False, True] 
                          for next_state in next_state_batch], dtype=np.float32)
        )
        
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.cat(tuple(state for state in next_state_batch))

        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            action_batch2 = action_batch2.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()

        current_prediction_batch = model(state_batch)
        next_prediction_batch = model2(next_state_batch)

#         y_batch = torch.cat(
#             tuple(reward if terminal else reward + opt.gamma * torch.max(prediction) for reward, terminal, prediction in
#                   zip(reward_batch, terminal_batch, next_prediction_batch)))
#         assert False, f"next_prediction_batch: {next_prediction_batch.shape}\taction2: { action_batch2.shape}\treward:{reward_batch.shape}\tterminal_batch:{terminal_batch}"
        y_batch = []
        for idx, term in enumerate( terminal_batch):
            if term:
                y_batch.append(reward_batch[idx])
            else:
                result = reward_batch[idx] + opt.gamma * next_prediction_batch[idx] @ action_batch2[idx]
                y_batch.append(result)
        y_batch=torch.cat(y_batch)
        
#         y_batch = torch.cat(
#             tuple(reward if terminal else reward + opt.gamma * ( prediction * action_batch2 ) for reward, terminal, prediction in
#                   zip(reward_batch, terminal_batch, next_prediction_batch)))

        q_value = torch.sum(current_prediction_batch * action_batch, dim=1)

        optimizer.zero_grad()
        # y_batch = y_batch.detach()
        loss = criterion(q_value, y_batch)
        loss.backward()
        optimizer.step()

        state = next_state
        
        if iter % 20 == 0:
            print(f"Loss: {loss.item()}\n")
        iter += 1
        
        if iter % 100 == 0:
            model2 = copy.deepcopy(model)
            model2.eval()
        
#         print("Iteration: {}/{}, Action: {}, Loss: {}, Epsilon {}, Reward: {}, Q-value: {}".format(
#             iter + 1,
#             opt.num_iters,
#             action,
#             loss,
#             epsilon, reward, torch.max(prediction)))
#         writer.add_scalar('Train/Loss', loss, iter)
#         writer.add_scalar('Train/Epsilon', epsilon, iter)
#         writer.add_scalar('Train/Reward', reward, iter)
#         writer.add_scalar('Train/Q-value', torch.max(prediction), iter)
        if (iter+1) % 1000000 == 0:
            torch.save(model, "{}/my_flappy_bird_{}".format(opt.saved_path, iter+1))
    torch.save(model, "{}/my_flappy_bird".format(opt.saved_path))


if __name__ == "__main__":
    opt = get_args()
    train(opt)
