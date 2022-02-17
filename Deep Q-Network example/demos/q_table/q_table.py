""" version 3 到這裡都還沒有使用NN
Q learning 是一個 實作強化學習的方法
這裡是使用 Q table (等同NN架構的作用) 時做出Qfunction
    就是帶入目前環境參數 透過table的方式 找出當下最好的決定 
    如果有更好的抉擇 就更新table對應的那個值
"""
"""
Agent learns the policy based on Q-learning with Q table.
Based on the example here: https://medium.com/@tuzzer/cart-pole-balancing-with-q-learning-b54c6068d947
"""
import math

import gym
import numpy as np
import time


def choose_action(state, q_table, action_space, epsilon):
    # state 已經轉成 我們定義的離散值了 所以可以直接用這個離散值 取相對應的table位置
    if np.random.random_sample() < epsilon: # random action
        return action_space.sample()
    else: # greddy action based on Q table
        # 應該要有取值 跟 如果reward比較好的話要覆寫 Q table 才對啊? 好像沒看到覆寫回去的步驟
        # 喔 有拉 在下面
        # 不懂為什麼要取第幾個 ? 第0個是當下環境做向左移動的評價 第1個是向右的
        return np.argmax(q_table[state]) 

def get_state(observation, n_buckets, state_bounds):
    state = [0] * len(observation) 
    for i, s in enumerate(observation):
        l, u = state_bounds[i][0], state_bounds[i][1] # lower- and upper-bounds for each feature in observation
        if s <= l:
            state[i] = 0
        elif s >= u:
            state[i] = n_buckets[i] - 1
        else:
            state[i] = int(((s - l) / (u - l)) * n_buckets[i])  
            # 這裡應該是把 observation 算成 0~n_buckets 分
            # 也就是 現在的 [position of cart, velocity of cart, angle of pole, rotation rate of pole] 的離散值

    return tuple(state)

if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    # Preparing Q table
    ## buckets for continuous state values to be assigned to
    n_buckets = (1, 1, 6, 3) # Observation space: [position of cart, velocity of cart, angle of pole, rotation rate of pole]
                             # Setting bucket size to 1 = ignoring the particular observation state 

    ## discrete actions
    n_actions = env.action_space.n  # 這裡就是2
    # print("n_actions : "+str(n_actions))

    ## state bounds
    # 感覺是弄出四個參數的界線
    state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
    print(state_bounds)
    # 如果 observation 已經定義好 為什麼這裡又要再調整 ??
    state_bounds[1] = [-0.5, 0.5]
    state_bounds[3] = [-math.radians(50), math.radians(50)]
    print(state_bounds)
    
    ## init Q table for each state-action pair
    ## 初始化 Q table 的空間  共五個維度 n_buckets(4個) + env.action_space.n(1個) # 雖然不知道定義
    q_table = np.zeros(n_buckets + (n_actions,))
    print("q_table : ")
    print(q_table)

    # Learning related constants; factors determined by trial-and-error
    get_epsilon = lambda i: max(0.01, min(1, 1.0   - math.log10((i+1)/25))) # epsilon-greedy, factor to explore randomly; discounted over time
    get_lr =      lambda i: max(0.01, min(0.5, 1.0 - math.log10((i+1)/25))) # learning rate; discounted over time
    gamma = 0.99 # reward discount factor

    # Q-learning
    for i_episode in range(200):
        epsilon = get_epsilon(i_episode)
        lr = get_lr(i_episode)

        observation = env.reset() # reset environment to initial state for each episode
        rewards = 0 # accumulate rewards for each episode
        state = get_state(observation, n_buckets, state_bounds) # turn observation into discrete state
        for t in range(250):
            env.render()

            # Agent takes action
            action = choose_action(state, q_table, env.action_space, epsilon) # choose an action based on q_table 
            # print("action : "+str(action))
            observation, reward, done, info = env.step(action) # do the action, get the reward
            rewards += reward
            next_state = get_state(observation, n_buckets, state_bounds)
            # print("next_state : "+str(next_state))

            # Agent learns via Q-learning
            # print("q_table[next_state] : "+str(q_table[next_state]))
            q_next_max = np.amax(q_table[next_state])
            # q_table 裡面存的是 當下環境 + 某個動作的 評價 (所以當下環境要做什麼動作比較好 是用左跟右的評價哪個比較好決定的)
            # 原本的值下降一點點(可以當作是上一次的結果 對下次的結果 影響力是 原本*gamma )
            q_table[state + (action,)] += lr * (reward + gamma * q_next_max - q_table[state + (action,)])

            # Transition to next state
            state = next_state

            if done:
                print("q_table : ")
                print(q_table)
                print('Episode finished after {} timesteps, total rewards {}'.format(t+1, rewards))
                break

    env.close() # need to close, or errors will be reported
