import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from tqdm import tqdm as _tqdm
import random


def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer


#@profile
def select_action(model, state, epsilon):
    with torch.no_grad():
        actions = model(torch.FloatTensor(state))
        argmax = torch.max(actions, 0)[1]
        n = actions.size(0)
        action = np.random.choice(n,1,
                    p = [epsilon / n + ((1-epsilon) if i == argmax else 0)
                             for i in range(n)
                        ])
    return action[0]

#@profile
def compute_q_val(model, state, action):
    actions = model(torch.FloatTensor(state))
    #     print(actions)
    return actions[range(len(state)), action]

#@profile
def compute_target(model, reward, next_state, done, discount_factor):
    # done is a boolean (vector) that indicates if next_state is terminal (episode is done)
    # YOUR CODE HERE
    Qvals = model(next_state)
    target = reward + discount_factor * Qvals.max(1)[0] * (1 - done.float())

    return target


#################################################################
###################### TRAIN FUNCTIONS  #########################
#################################################################
#@profile
def train_QNet(model, memory, optimizer, batch_size, discount_factor):
    # DO NOT MODIFY THIS FUNCTION

    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)

    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)

    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)
    done = torch.tensor(done, dtype=torch.uint8)  # Boolean

    # compute the q value
    q_val = compute_q_val(model, state, action)

    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_target(model, reward, next_state, done, discount_factor)

    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())



j = 0
#@profile
def train_QNet_true_gradient(model, memory, optimizer, batch_size, discount_factor,
                             target_model=None):
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)

    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)

    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)
    done = torch.tensor(done, dtype=torch.uint8)  # Boolean

    # compute the q value
    q_val = compute_q_val(model, state, action)

    #     with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
    target_model = model if target_model is None else target_model
    target = compute_target(target_model, reward, next_state, done, discount_factor)

    # loss is measured from error between current and newly expected Q values

    # we can do this as the smooth l1 is symmetric
    loss = F.smooth_l1_loss(target - q_val, torch.zeros_like(q_val))

    # alternative: calculate gradient two times for both variables
    #     loss = F.smooth_l1_loss(q_val, target.detach())
    #     loss2 = F.smooth_l1_loss(target, q_val.detach())
    #     loss = loss+loss2

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())
