"""
Agent learns the policy based on Q-learning with Deep Q-Network.
Based on the example here: https://morvanzhou.github.io/tutorials/machine-learning/torch/4-05-DQN/
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import time

"""
這個方法其實有點像 AB test
比較好就覆寫原本使用的方法
如何比較 AB 兩個 test 呢?
我們自己定義了 reward
所以把reward的分數加總 就可以比較了

這裡有一個問題?
不是每一次的結果都是用 NN 算出來的啊?
這樣這個結果可以加總嗎?
"""
# Cheating mode speeds up the training process
CHEAT = True


# Basic Q-netowrk
class NN_Net(nn.Module):
    def __init__(self, n_states, n_actions, n_hidden):
        super(NN_Net, self).__init__()

        # Two fully-connected layers, input (state) to hidden & hidden to output (action)
        self.fc1 = nn.Linear(n_states, n_hidden)  #(n_states() * n_hidden 個權重)
        self.out = nn.Linear(n_hidden, n_actions) #(n_hidden * n_actions 個權重)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


# Deep Q-Network, composed of one eval network, one target network
class DQN(object):
    def __init__(self, n_states, n_actions, n_hidden, batch_size, lr, epsilon, gamma, target_replace_iter, memory_capacity):
        # <<跟 NN 有關的東西的設定>>
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_hidden = n_hidden
        # eval_net 丟進去 state 會產出 各個action的結果的預期評價
        self.eval_net, self.target_net = NN_Net(n_states, n_actions, n_hidden), NN_Net(n_states, n_actions, n_hidden)
        # 計算 loss 的方法
        self.loss_func = nn.MSELoss()
        # 計算梯度之後所使用的優化方法
        # 注意這裡有帶入eval_net 所以這個optimizer只會更新eval_net 
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)

        
        # training 相關
        self.lr = lr #就是learning rate
        self.batch_size = batch_size
        self.learn_step_counter = 0 # 單純計數 現在跑了幾次
        self.target_replace_iter = target_replace_iter # 根據上面的計數 設定幾次訓練之後把資料寫回
        self.gamma = gamma # 遺忘速率
        
        # <<跟記錄有關>>
        # 用來記錄每一次的結果 會拿這些結果進行 training
        self.memory_capacity = memory_capacity # 紀錄最大儲存量
        self.memory = np.zeros((memory_capacity, n_states * 2 + 2)) # initialize memory, each memory slot is of size (state + next state + reward + action)
        self.memory_counter = 0
        self.epsilon = epsilon # 要選擇隨機動作還是使用NN的機率 為了取得比較多樣的紀錄

    def choose_action(self, state):
        x = torch.unsqueeze(torch.FloatTensor(state), 0)

        # epsilon-greedy
        if np.random.uniform() < self.epsilon: # random
            action = np.random.randint(0, self.n_actions)
        else: # greedy
            actions_value = self.eval_net(x) # feed into eval net, get scores for each action
            action = torch.max(actions_value, 1)[1].data.numpy()[0] # choose the one with the largest score

        return action

    # 把參數這些傳進來的值 存起來
    def store_transition(self, state, action, reward, next_state):
        # Pack the experience
        transition = np.hstack((state, [action, reward], next_state))

        # Replace the old memory with new memory
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # Randomly select a batch of memory to learn from
        # 隨機選 batch_size 個數字 最大值為 memory_capacity(數字會重複) 
        # 為什麼不直接用最新的一次訓練就好? 因為每次訓練不只使用一次紀錄
        sample_index = np.random.choice(self.memory_capacity, self.batch_size)
        # print("in learn sample_index : "+str(sample_index))
        b_memory = self.memory[sample_index, :]
        # b_memory 是 batch_size 個 memory 所以才需要把紀錄存到 memory 裡面
        # print("in learn b_memory : "+str(len(b_memory)))

        # 整形
        b_state = torch.FloatTensor(b_memory[:, :self.n_states])
        b_action = torch.LongTensor(b_memory[:, self.n_states:self.n_states+1].astype(int))
        b_reward = torch.FloatTensor(b_memory[:, self.n_states+1:self.n_states+2])
        b_next_state = torch.FloatTensor(b_memory[:, -self.n_states:])

        # Compute loss between Q values of eval net & target net
        # 這樣不算標記嗎? 因為這種方法是直接給判斷結果好壞的規則 要NN算出這些state相對應的動作怎樣算好 的確沒有給label
        # 要查 gather 的作用? 就是整形 才可以放得進去
        # 所以丟一個list進去 會一個一個丟進去 回傳結果的list ? 應該會
        q_eval = self.eval_net(b_state).gather(1, b_action) # evaluate the Q values of the experiences, given the states & actions taken at that time
        # 不知道 detach 定義? detach的確只是算優化 detach出來的NN 不能進行backward
        # 照理來說也要gather才對
        q_next = self.target_net(b_next_state).detach() # detach from graph, don't backpropagate
        print("type :",type(q_next))
        # print("q_next :",q_next)
        # print("q_next.max(1) :",q_next.max(1))
        # .max 是把reward結果比較好的那一個抓出來
        q_target = b_reward + self.gamma * q_next.max(1)[0].view(self.batch_size, 1) # compute the target Q values
        print("q_target :",q_target)
        print("q_eval :",q_eval)
        print("q_eval.size() :",q_eval.size())  # torch.Size([32, 1])
        print("q_target.size() :",q_target.size())  # torch.Size([32, 1])

        loss = self.loss_func(q_eval, q_target) 
        # 這個只有一個value 為什麼可以backward ?  
        # 因為loss只是一個數字而已 我們是用loss這個數字進行backward 所以我們是用q_eval,q_target去計算loss這個數字
        print("loss :", loss)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network every few iterations (target_replace_iter), i.e. replace target net with eval net
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_replace_iter == 0:
            # 把 eval_net 複製 更新 target_net
            self.target_net.load_state_dict(self.eval_net.state_dict())


if __name__ == '__main__':
    print_log = False
    
    env = gym.make('CartPole-v0')
    env = env.unwrapped # For cheating mode to access values hidden in the environment

    # Environment parameters
    n_actions = env.action_space.n  # 有幾種動作可以做
    print("how many actions can do :", n_actions)  # 只有左跟右
    n_states = env.observation_space.shape[0]  # 有幾個資訊 
    # 其實不需要知道這幾個資訊代表什麼 反正就是資訊 丟進去NN裡面 讓NN找什麼動作最好的規律
    print("how many informations :", n_states)

    # Hyper parameters
    n_hidden = 50
    batch_size = 32
    lr = 0.01                 # learning rate
    epsilon = 0.1             # epsilon-greedy, factor to explore randomly
    gamma = 0.9               # reward discount factor
    target_replace_iter = 100 # target network update frequency
    memory_capacity = 100#2000
    n_episodes = 400 if CHEAT else 4000

    # Create DQN
    dqn = DQN(n_states, n_actions, n_hidden, batch_size, lr, epsilon, gamma, target_replace_iter, memory_capacity)

    # Collect experience
    for i_episode in range(n_episodes):
        t = 0 # timestep
        rewards = 0 # accumulate rewards for each episode
        state = env.reset() # reset environment to initial state for each episode
        # 我覺得這邊不應該用 while true 因為這樣大角度的可能都沒訓練到 只訓練到小角度 因為到最後基本上是站著的 
        # 不過這樣是可以測試結果到底有沒有優化 可以看到底可以站多久
        while True:  
            env.render()

            # Agent takes action
            action = dqn.choose_action(state) # choose an action based on DQN
            next_state, reward, done, _info = env.step(action) # do the action, get the reward

            # Cheating part: modify the reward to speed up training process
            if CHEAT:
                x, v, theta, omega = next_state
                r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8 # reward 1: the closer the cart is to the center, the better
                r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5 # reward 2: the closer the pole is to the center, the better
                # 使用 next_state 重新計算 reward (跟棒子角度與距離中間的距離有關)
                # 如果只是要棒子站立 是不是只需要計算 r2 就好 ??
                reward = r1 + r2

            # Keep the experience in memory
            dqn.store_transition(state, action, reward, next_state)

            # Accumulate reward
            rewards += reward

            # If enough memory stored, agent learns from them via Q-learning
            if dqn.memory_counter > memory_capacity:
                print_log = True
                dqn.learn() 
                time.sleep(5)          

            # Transition to next state
            state = next_state

            if done:
                print('Episode finished after {} timesteps, total rewards {}'.format(t+1, rewards))
                break

            t += 1

    env.close()