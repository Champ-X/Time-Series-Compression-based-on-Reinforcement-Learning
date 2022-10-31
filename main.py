import tensorflow as tf
import random
import datetime
import math
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten,Embedding,Dropout
from keras.optimizers import rmsprop_v2
from keras.models import load_model
from collections import deque
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from Env import Env
from preprocess import cut_data
import pickle

"""
class Maze(object):
    def __init__(self, size=10, blocks_rate=0.1):
        self.size = size if size > 3 else 10
        self.blocks = int((size ** 2) * blocks_rate)
        self.s_list = []
        self.maze_list = []
        self.e_list = []

    def create_mid_lines(self, k):
        if k == 0:
            self.maze_list.append(self.s_list)
        elif k == self.size - 1:
            self.maze_list.append(self.e_list)
        else:
            tmp_list = []
            for l in range(0, self.size):
                if l == 0:
                    tmp_list.extend("#")
                elif l == self.size - 1:
                    tmp_list.extend("#")
                else:
                    a = random.randint(-1, 0)
                    tmp_list.extend([a])
            self.maze_list.append(tmp_list)

    def insert_blocks(self, k, s_r, e_r):
        b_y = random.randint(1, self.size - 2)
        b_x = random.randint(1, self.size - 2)
        if [b_y, b_x] == [1, s_r] or [b_y, b_x] == [self.size - 2, e_r]:
            k = k - 1
        else:
            self.maze_list[b_y][b_x] = "#"

    def generate_maze(self):
        s_r = random.randint(1, int((self.size / 2)) - 1)
        for i in range(0, self.size):
            if i == s_r:
                self.s_list.extend("S")
            else:
                self.s_list.extend("#")
        start_point = [0, s_r]

        e_r = random.randint(int((self.size / 2)) + 1, self.size - 2)
        for j in range(0, self.size):
            if j == e_r:
                self.e_list.extend([50])
            else:
                self.e_list.extend("#")
        goal_point = [self.size - 1, e_r]

        for k in range(0, self.size):
            self.create_mid_lines(k)

        for k in range(self.blocks):
            self.insert_blocks(k, s_r, e_r)

        return self.maze_list, start_point, goal_point

class Field(object):
    def __init__(self, maze, start_point, goal_point):
        self.maze = maze
        self.start_point = start_point
        self.goal_point = goal_point
        self.movable_vec = [[1,0],[-1,0],[0,1],[0,-1]]

    def display(self, point=None):
        field_data = copy.deepcopy(self.maze)
        if not point is None:
                y, x = point
                field_data[y][x] = "@@"
        else:
                point = ""
        for line in field_data:
                print ("\t" + "%3s " * len(line) % tuple(line))

    def get_actions(self, state):
        movables = []
        if state == self.start_point:
            y = state[0] + 1
            x = state[1]
            a = [[y, x]]
            return a
        else:
            for v in self.movable_vec:
                y = state[0] + v[0]
                x = state[1] + v[1]
                if not(0 < x < len(self.maze) and
                       0 <= y <= len(self.maze) - 1 and
                       maze[y][x] != "#" and
                       maze[y][x] != "S"):
                    continue
                movables.append([y,x])
            if len(movables) != 0:
                return movables
            else:
                return None

    def get_val(self, state):
        y, x = state
        if state == self.start_point: return 0, False
        else:
            v = float(self.maze[y][x])
            if state == self.goal_point:
                return v, True
            else:
                return v, False

"""

loss =[]
rewardd=[]

class DQN_Solver:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.gamma = 0.9
        self.epsilon = 1.0
        self.e_decay = 0.9999
        self.e_min = 0.01
        self.learning_rate = 0.0001
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(256, input_shape=(2,4), activation='tanh'))
        model.add(Flatten())
        model.add(Dense(256, activation='tanh'))
        model.add(Dense(256, activation='tanh'))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='linear'))
        model.compile(loss="mse", optimizer=rmsprop_v2.RMSprop(lr=self.learning_rate))
        return model

    def remember_memory(self, state,value, action, reward, next_state,next_value, next_movables, done):
        self.memory.append((state,value, action, reward, next_state,next_value, next_movables, done))

    def choose_action(self, state, movables,value):
        if self.epsilon >= random.random():
            return random.choice(movables)
        else:
            return self.choose_best_action(state, movables,value)

    def cal_charact(self,state,value):
        length = len(state)
        sum_all = 0
        sum_ = 0
        sum_res = 0
        sum_half = 0
        i=0
        for item in state:
            d = len(item)
            count = 0
            for e in item:
                if e=='#':
                    sum_=sum_+1
                    count=count+1
                elif e=='1':
                    sum_all=sum_all+2
                else:
                    sum_all=sum_all+1
            sum_res=sum_*value[i]+sum_res
            i=i+1
            if 2*count<=len(item):
                sum_half=sum_half+1
        #print(i)
        #print(sum_res)
        return [length,sum_all,sum_res/length,sum_half]


    def choose_best_action(self, state, movables,value):
        best_actions = []
        max_act_value = -100
        for a in movables:
            temp = self.cal_charact(state,value)
            np_action=np.array([[temp,[a,0,0,0]]])
            act_value = self.model.predict(np_action)
            #print(act_value)
            if act_value > max_act_value:
                best_actions = [a, ]
                max_act_value = act_value
            elif act_value == max_act_value:
                best_actions.append(a)
        return random.choice(best_actions)

    def replay_experience(self, batch_size):
        batch_size = min(batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)
        X = []
        Y = []
        for i in range(batch_size):
            state, value,action, reward, next_state,next_value, next_movables, done = minibatch[i]
            t = self.cal_charact(state,value)
            tt=[t,[action,0,0,0]]
            if done:
                target_f = reward
            else:
                next_rewards = []
                for i in next_movables:
                    temp=self.cal_charact(next_state,next_value)
                    np_next_s_a = np.array([[temp,[i,0,0,0]]])
                    #e= tf.keras.layers.Embedding(input_dim=flag*2+1, output_dim=1)(np_next_s_a)
                    next_rewards.append(self.model.predict(np_next_s_a))
                np_n_r_max = np.amax(np.array(next_rewards))
                target_f = reward + self.gamma * np_n_r_max
            X.append(tt)
            Y.append(target_f)
        np_X = np.array(X)
        np_Y = np.array([Y]).T
        #self.model.fit(np_X, np_Y, epochs=5, verbose=0)
        history = self.model.fit(np_X, np_Y, epochs=1, verbose=0, validation_split=0.2)  # 获取数据
        temp = history.history['loss']
        loss.append(temp)



        if self.epsilon > self.e_min:
            self.epsilon *= self.e_decay

    def save_model(self):
        self.model.save(('my_model.h5'))

    def load_model(self):
        model = load_model('my_model1000.h5')
        self.model=model


# 画图
def draw_pic(loss):
    loss = list(np.ravel(loss))
    x1 = range(1000)
    y1 = loss
    plt.plot(x1, y1, label="Train_Loss")
    plt.show()
    plt.savefig("picture.png",dpi=500)




if __name__ == "__main__":
    files = ['data/mit101.txt']
    for item in files:
        l, data, test0, value = cut_data(item)
        # data=['1001',"1000","1110","1100"]
        env = Env(data)
        env.reset()

        state_size = 4
        action_size = 5
        dql_solver = DQN_Solver(state_size, action_size)
        """
    ###train
        episodes = 300
        times = 100
        for e in range(episodes):
            state=env.reset()
            #print('state:',state)
            score=0
            for time in range(times):
                #print(time)
                movables = env.get_actions(state)
                #print('move:')
                #print(movables)
                action = dql_solver.choose_action(state, movables,list(state.values()))
                #print(action)
                base, reward, done = env.gostep(action)
                score = score + reward
                next_state = base
                next_movables = env.get_actions(next_state)
                #print('nextmove:')
                #print(next_movables)
                dql_solver.remember_memory(list(state),list(state.values()), action, reward, list(next_state),list(next_state.values()), next_movables, done)
                if done or time == (times - 1):
                    if e % 5 == 0:
                        print("episode: {}/{}, score: {}, e: {:.2} \t @ {}"
                              .format(e, episodes, score, dql_solver.epsilon, time))
                    break
                if time%10 ==0:
                    base = env.change_base()
                state = base
            print(len(state))
            rewardd.append(score)
            dql_solver.replay_experience(32)
        dql_solver.save_model()
        #draw_pic(loss)
        with open('result.txt','w',encoding='utf-8')as file:
            file.write(str(loss)+'\n')
            file.write(str(rewardd))

    """

        # predict
        steps = 0
        test = value
        env1 = Env(test)
        start = 0
        state = env1.reset()
        for item in test:
            start = len(item) + start

        dql_solver.load_model()
        time_start = datetime.datetime.now()
        while True:
            steps += 1
            movables = env1.get_actions(state)
            action = dql_solver.choose_best_action(state, movables, list(state.values()))
            # print("current state: {0} -> action: {1} ".format(state, action))
            base, reward, done = env1.gostep(action)
            if (steps + 1) % 20 == 0:
                base = env1.change_base()
            state = base
            # print("current step: {0} \t score: {1}\n".format(steps, reward))
            if done:
                end = datetime.datetime.now()
                print('time: {0}'.format(end - time_start))
                r = 0
                state = dict(sorted(state.items(), key=lambda x: x[1], reverse=True))
                flag = 0
                for key, value in state.items():
                    if value == 1:
                        r = r + len(key)
                    else:
                        score1 = key.count('#') * value + len(bin(flag)[2:]) * value
                        # score2 = math.log(len(state), 2) * len(state)
                        r = r + score1 + len(key.replace('#', ''))
                       # r = r + score1
                        flag += 1
                print(r)
                print(len(state))
                print(start)
                print("goal!")
                print(start / r)
                break




