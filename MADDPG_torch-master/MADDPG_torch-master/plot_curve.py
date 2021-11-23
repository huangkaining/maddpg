import json
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

exp_name = "simple_world_comm_missile_21_1121_2"
model_file_dir = os.path.join(r'./models-pytorch',exp_name,"data.txt")
data_all = []
with open(model_file_dir,'r') as f:
    for i in f:
        data = json.loads(i)
        data_all.append(data)

x = []
y = []
agent_reward = [[] for _ in range(4)]
for dic in data_all:
    r = dic['episode_gone']
    #if r>1500: break
    ydata = dic['episode_rewards']
    x.extend(list(range(r-100,r)))
    y.extend(ydata)
    for i in range(4):
        agent_reward[i].extend(dic[str(i)])
plt.subplot(3,2,1)
plt.title("Reward-Episode")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.plot(x,y)
plt.subplot(3,2,2)
plt.title("Average Reward-Episode")
plt.xlabel("Episode")
plt.ylabel("Reward")
smooth_rate = 200
y_avg = []
for i in range(len(y)-smooth_rate-1):
    y_avg.append(np.mean(y[i:i+smooth_rate]))
plt.plot(list(range(len(y_avg))),y_avg)

cc = 1

for i in range(4):
    plt.subplot(3,2,3 + i)
    plt.title("agent " + str(i))
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    if cc == 1:
        agent_reward_avg = []
        for p in range(len(agent_reward[i]) - smooth_rate - 1):
            agent_reward_avg.append(np.mean(agent_reward[i][p:p + smooth_rate]))
        plt.plot(list(range(len(agent_reward_avg))), agent_reward_avg)
    else:
        plt.plot(x,agent_reward[i])
'''for i in range(4):
    print(len(agent_reward[i]))'''
plt.show()