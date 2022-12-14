import random
import math
from sklearn.cluster import KMeans
import numpy as np
import gc
from query import query1
from query import query2
from query import query3
def match(str, pattern):
    if set(str)=='#':
        return False
    if str == '' and (pattern == '' or set(pattern) == {'#'}):
        return True
    if str == '' and (pattern != '' or set(pattern) != {'#'}):
        return False
    if str != '' and set(pattern) == {'#'}:
        return True
    if pattern[0:1] != '#':
        if str[0:1] == pattern[0:1]:
            return match(str[1:], pattern[1:])
        else:
            return False
    else:
            return match(str[1:], pattern[1:])

class Env():
    def __init__(self, data):
        self.action_space = ['d0','d1' 'a0', 'a1']#0:whole,1:part
        self.n_actions = len(self.action_space)
        self.data = data
        self.base = {}
        self.value = 150
        self.step = 0
        self.result = []
        self.index = {}

    def reset(self):
        self.base={}
        self.result=[]
        self.index={}
        temp ={}
        data = self.data
        i = 0
        for item in data:
            temp[item] = ''
            self.result.append(temp)
            if item not in self.base.keys():
                self.base[item]=1
                self.index[item] = [i]
            else:
                self.base[item] +=1
                self.index[item].append(i)
            i = i+1
            temp = {}
        return self.base

    def find_match(self,base):
        temp={}
        result={}
        left_data=[]
        for key in base:
            temp[key]=len(key.replace('#',''))
            result[key]=0
        temp = dict(sorted(temp.items(), key=lambda x: x[1], reverse=True))
        for data in self.data:
            flag=0
            for key in temp.keys():
                if match(data,key)==True:
                    result[key] += 1
                    break
                else:
                    flag+=1
            if flag==len(temp):
                left_data.append(data)
        for k in list(result.keys()):
            if result[k]==0:
                del result[k]
        for item in left_data:
            result[item]=1
        gc.collect()
        return result

    def find_result(self,data,base):
        st =''
        if len(data)!=len(base):
            print('error')
            print(data)
            print(base)
            return 0
        for i in range(len(base)):
            if base[i]=='#':
                st=st+data[i]
        return st

    def renew(self,new,base):
        print('renew')
        t = base
        for k in t.keys():
            t[k]=len(k.replace('#',''))
        t = dict(sorted(t.items(), key=lambda x: x[1], reverse=True))
        #print(t)
        for item in new:
            for b in t.keys():
                if item ==b :
                    break
                if len(item)!=len(b):
                    continue
                if match(b,item) ==True:
                    temp = self.index[item]
                    for i in temp:
                        if match(self.data[i],b):
                            self.base[item]=self.base[item]-1
                            self.base[b]=self.base[b]+1
                            self.index[item].remove(i)
                            self.index[b].append(i)
                            self.result[i]={}
                            self.result[i][b]=self.find_result(self.data[i],b)


    def change_base(self):
        b={}
        for k,v in self.base.items():
            b[k]=v
        for key, value in self.base.items():
            if value==1:
                temp = self.index[key][0]
                self.index.pop(key)
                self.result[temp]={}
                data = self.data[temp]
                self.result[temp][data]=''
                b.pop(key)
                b[data]=1
                self.index[data]=[temp]
            else:
                continue
        self.base =b

        return self.base

    # ???kmeans???#?????????#??????????????????
    def find_midpoint(self,base):
        temp = np.array(list(base.values()))
        if len(base)<2 or len(set(temp))==1:
            return 1
        kmeans = KMeans(n_clusters=2)  # n_clusters:number of cluster
        temp = temp.reshape(-1,1)
        if kmeans.fit(temp):
            flag = kmeans.labels_[0]
            for i in range(len(kmeans.labels_) - 1):
                if kmeans.labels_[i + 1] != flag:

                    return i + 1

            return 1
        else:

            return 1

    #??????#?????????dic
    def find_changeable(self, base):
        result = {}
        for key, value in base.items():
            #??????#?????????
            x = key.count('#')
            a = len(key)
            k = math.log(len(base),2)
            # ????????????base?????????????????????
            if a*value <= a +value*x + value*k:
                result[key]=x
        return result


    def get_actions(self, state):
        #?????????????????????
        change = self.find_changeable(state)
        #??????#????????????
        v = dict(sorted(change.items(), key=lambda x: x[1], reverse=False))
        l = list(v.keys())
        #??????????????????????????????base???
        mid = self.find_midpoint(v)
        #????????????
        if mid==1 or mid ==len(v)-1:
            if l[0].count('#')<int(len(l[0])/3):
                return [0, 1]
            else:
                return [2,3]
        temp = int(len(v)/2)
        if mid >=temp:
            return [0,1]
        else:
            return [2,3]


        """
        # 01?????????base
        flag0 = 0
        # #?????????base
        flag1 = 0
        for key, value in state.items():
            if "#" not in key:
                flag0 += 1
            if '0' not in key and '1' not in key:
                flag1 += 1
        if flag0 == len(state) and flag1 ==0:
            return [0, 1]
        if flag1 !=0:
            return [2,3]
        elif flag0 != 0 and flag1 == 0:
            if flag0>=int(len(state)/2):
                return [0, 1]
            else:
                return [0,1,3]
        elif flag0 == 0 and flag1 == 0:
            return [0, 1, 2, 3]
        elif flag0 != 0 and flag1 != 0:
            return [1, 3]
        elif flag0 == 0 and flag1 != 0:
            return [1, 2, 3]
            """

    def gostep(self, action):
        base = self.base
        start_score = 0
        end_score = 0
        flag = 0
        for key,value in base.items():
            if value == 1:
                start_score =start_score+len(key)
            score1 = key.count('#') * value + len(bin(flag)[2:]) * value
            flag += 1
            # score2 = math.log(len(self.base), 2) * len(self.base)
            start_score = score1 + start_score+len(key)
        #start_score=start_score+len(base)*len(list(base.keys())[0])
        space = []
        base_space = []
        self.step += 1
        changeable = self.find_changeable(base)
        #??????????????????????????????????????????
        if len(changeable)==0:
            return self.base,0, True
        #print('changeable:')
        #print(len(changeable))
        v = dict(sorted(changeable.items(), key=lambda x: x[1], reverse=False))
        if action == 0:  # delete ??????
            #base_value = self.value
            #v= dict(sorted(base.items(), key=lambda x: x[1], reverse=False)[:base_value])
            #v = random.sample(list(base), int(len(base)/2))
            #??????#?????????????????????#????????????????????????
            midpoint = self.find_midpoint(v)
            if midpoint == 1 or midpoint ==len(v)-1:
                t = list(v)
            else:
                t=list(v)[:midpoint]
            combine = []
            use= []
            use_point = 0
            for item in t:
                if item in combine:
                    continue
                if '0' not in item and '1' not in item:
                    continue
                for i in range(len(item)):
                    if item[i] != "#":
                        space.append(i)
                    elif item[i] ==".":
                        continue
                    else:
                        use.append(i)#??????#?????????
                if len(space) == 1:
                    temp0 = 0
                else:
                    temp0 = random.randint(0, len(space) - 1)
                #??????????????????
                #temp0 = int(len(space)/2)
                temp = space[temp0]#??????item?????????
                if len(use)!=0:
                    if len(use)==1:
                        if temp>use[0]:
                            use_point =1
                        else:
                            use_point = 0
                    else:
                        for l in range(len(use)):
                            if temp < use[l]:
                                use_point = l
                                break
                            use_point=len(use)

                space = []
                use = []
                new = item[:temp] + '#' + item[temp + 1:]
                base_space.append(new)
                old_data = self.index[item]
                if new in base.keys() and new not in t:
                    for i in old_data:
                        self.index[new].append(i)
                        st = self.result[i][item]
                        st = st[:use_point] + item[temp] +st[use_point:]
                        self.result[i]={}
                        self.result[i][new]=st
                    self.base[new] = self.base[new] + len(old_data)
                elif new not in base.keys():
                    self.index[new] = []
                    for i in old_data:
                        self.index[new].append(i)
                        st = str(self.result[i][item])

                        st = st[:use_point] + item[temp] +st[use_point:]
                        self.result[i] = {}
                        self.result[i][new]=st
                    self.base[new] = len(old_data)
                else:
                    combine.append(new)
                    old_data = self.index[item]
                    for i in old_data:
                        self.index[new].append(i)
                        st = self.result[i][item]
                        st = st[:use_point] + item[temp] + st[use_point:]
                        self.result[i] = {}
                        self.result[i][new] = st
                    self.base[new] = self.base[new] + len(old_data)
                self.index.pop(item)
                self.base.pop(item)

            #print(self.index)
            #print(self.base)
            #print(self.result)

        if action == 1:  # delete last bit
            midpoint = self.find_midpoint(v)
            if midpoint==1:
                t=list(v)
            else:
                t=list(v)[:midpoint]
            combine = []
            use = []
            use_point = 0
            for item in t:
                if item in combine:
                    continue
                if '0' not in item and '1' not in item:
                    continue
                for i in range(len(item)):
                    if item[i] != "#":
                        space.append(i)
                    elif item[i] ==".":
                        continue
                    else:
                        use.append(i)  # ??????#?????????
                # ???????????????
                temp = space[-1]
                if len(use) != 0:
                    if len(use) == 1:
                        if temp > use[0]:
                            use_point = 1
                        else:
                            use_point = 0
                    else:
                        for l in range(len(use)):
                            if temp < use[l]:
                                use_point = l
                                break
                            use_point = len(use)
                space = []
                use = []
                new = item[:temp] + '#' + item[temp + 1:]
                base_space.append(new)
                if new in base.keys() and new not in t:
                    old_data = self.index[item]
                    for i in old_data:
                        self.index[new].append(i)
                        st = self.result[i][item]
                        st = st[:use_point] + item[temp] + st[use_point:]
                        self.result[i] = {}
                        self.result[i][new] = st
                    self.base[new] = self.base[new] + len(old_data)
                elif new not in base.keys():
                    old_data = self.index[item]
                    self.index[new]=[]
                    for i in old_data:
                        self.index[new].append(i)
                        st = self.result[i][item]
                        st = st[:use_point] + item[temp] + st[use_point:]
                        self.result[i] = {}
                        self.result[i][new] = st
                    self.base[new] = len(old_data)
                else:
                    combine.append(new)
                    old_data = self.index[item]
                    for i in old_data:
                        self.index[new].append(i)
                        st = self.result[i][item]
                        st = st[:use_point] + item[temp] + st[use_point:]
                        self.result[i] = {}
                        self.result[i][new] = st
                    self.base[new] = self.base[new] + len(old_data)
                self.index.pop(item)
                self.base.pop(item)
            #print(self.index)
            #print(self.base)
            #print(self.result)

        if action == 2:  # add ??????
            midpoint = self.find_midpoint(v)
            if midpoint == len(v)-1 :
                t = list(v)
            else:
                t=list(v)[midpoint:]

            if len(t)==0:
                t = v
            combine = []
            use = []
            for item in t:
                if "#" not in item:
                    continue
                for i in range(len(item)):
                    if item[i] == "#":
                        space.append(i)
                    else:
                        use.append(i)  # ???#?????????
                if len(space) == 1:
                    temp0 = 0
                else:
                    temp0 = random.randint(0, len(space) - 1)
                # ??????????????????
                temp = space[temp0]
                # ?????????????????????
                space = []
                add = random.randint(0, 1)
                new = item[:temp] + str(add) + item[temp + 1:]#??????base
                new_data=[]
                son = item[:temp] + str(1-add) + item[temp + 1:]#???base??????base?????????base
                son_data = []
                old_data = self.index[item]
                for k in old_data:
                    if str(self.result[k][item])[temp]==str(add):
                        new_data.append(k)
                    else:
                        son_data.append(k)
                if len(new_data)!=0:
                    base_space.append(new)
                    if new in base.keys() and new not in t:
                        for i in new_data:
                            self.index[new].append(i)
                            st = self.result[i][item]
                            st = st[:temp] + st[temp + 1:]
                            self.result[i] = {}
                            self.result[i][new] = st
                        self.base[new] = self.base[new] + len(new_data)
                    elif new not in base.keys():
                        self.index[new]=[]
                        for i in new_data:
                            self.index[new].append(i)
                            st = self.result[i][item]
                            st = st[:temp] + st[temp + 1:]
                            self.result[i] = {}
                            self.result[i][new] = st
                        self.base[new] = len(new_data)
                    else:
                        combine.append(new)
                        for i in new_data:
                            self.index[new].append(i)
                            st = self.result[i][item]
                            st = st[:temp] + st[temp + 1:]
                            self.result[i] = {}
                            self.result[i][new] = st
                        self.base[new] = self.base[new] + len(new_data)

                if len(son_data)!=0:
                    base_space.append(son)
                    if son in base.keys() and son not in t:
                        for i in son_data:
                            self.index[son].append(i)
                            st = self.result[i][item]
                            st = st[:temp] + st[temp + 1:]
                            self.result[i] = {}
                            self.result[i][son] = st
                        self.base[son] = self.base[son] + len(son_data)
                    elif son not in base.keys():
                        self.index[son] = []
                        for i in son_data:
                            self.index[son].append(i)
                            st = self.result[i][item]

                            st = st[:temp] + st[temp + 1:]
                            self.result[i] = {}
                            self.result[i][son] = st
                        self.base[son] = len(son_data)
                    else:
                        combine.append(son)
                        for i in son_data:
                            self.index[son].append(i)
                            st = self.result[i][item]
                            st = st[:temp] + st[temp + 1:]
                            self.result[i] = {}
                            self.result[i][son] = st
                        self.base[son] = self.base[son] + len(son_data)
                self.index.pop(item)
                self.base.pop(item)

            #print(self.index)
            #print(self.base)
            #print(self.result)

        if action == 3:  # add first bit
            midpoint = self.find_midpoint(v)
            if midpoint == len(v) - 1:
                t = list(v)
            else:
                t = list(v)[midpoint:]
            if len(t)==0:
                t = v
            combine = []
            use = []
            for item in t:
                if "#" not in item:
                    continue
                for i in range(len(item)):
                    if item[i] == "#":
                        space.append(i)
                    else:
                        use.append(i)  # ???#?????????
                temp0=0
                # ???????????????
                temp = space[temp0]
                # ?????????????????????
                space = []
                use = []
                add = random.randint(0, 1)
                new = item[:temp] + str(add) + item[temp + 1:]  # ??????base
                new_data = []
                son = item[:temp] + str(1 - add) + item[temp + 1:]  # ???base??????base?????????base
                son_data = []
                old_data = self.index[item]
                for k in old_data:
                    if str(self.result[k][item])[temp] == str(add):
                        new_data.append(k)
                    else:
                        son_data.append(k)
                if len(new_data) != 0:
                    base_space.append(new)
                    if new in base.keys() and new not in t:
                        for i in new_data:
                            self.index[new].append(i)
                            st = self.result[i][item]
                            st = st[:temp] + st[temp + 1:]
                            self.result[i] = {}
                            self.result[i][new] = st
                        self.base[new] = self.base[new] + len(new_data)
                    elif new not in base.keys():
                        self.index[new] = []
                        for i in new_data:
                            self.index[new].append(i)
                            st = self.result[i][item]
                            st = st[:temp] + st[temp + 1:]
                            self.result[i] = {}
                            self.result[i][new] = st
                        self.base[new] = len(new_data)
                    else:
                        combine.append(new)
                        for i in new_data:
                            self.index[new].append(i)
                            st = self.result[i][item]
                            st = st[:temp] + st[temp + 1:]
                            self.result[i] = {}
                            self.result[i][new] = st
                        self.base[new] = self.base[new] + len(new_data)

                if len(son_data) != 0:
                    base_space.append(son)
                    if son in base.keys() and son not in t:
                        for i in son_data:
                            self.index[son].append(i)
                            st = self.result[i][item]
                            st = st[:temp] + st[temp + 1:]
                            self.result[i] = {}
                            self.result[i][son] = st
                        self.base[son] = self.base[son] + len(son_data)
                    elif son not in base.keys():
                        self.index[son] = []
                        for i in son_data:
                            self.index[son].append(i)
                            st = self.result[i][item]

                            st = st[:temp] + st[temp + 1:]
                            self.result[i] = {}
                            self.result[i][son] = st
                        self.base[son] = len(son_data)
                    else:
                        combine.append(son)
                        for i in son_data:
                            self.index[son].append(i)
                            st = self.result[i][item]
                            st = st[:temp] + st[temp + 1:]
                            self.result[i] = {}
                            self.result[i][son] = st
                        self.base[son] = self.base[son] + len(son_data)
                self.index.pop(item)
                self.base.pop(item)

            #print(self.index)
            #print(self.base)
            #print(self.result)
        #????????????
        #self.renew(base_space,self.base)

        self.base = dict(sorted(self.base.items(), key=lambda x: x[1], reverse=True))
        flag = 0
        for key, value in self.base.items():
            if value == 1:
                end_score = end_score+len(key)
                continue
            score1 = key.count('#') * value + len(bin(flag)[2:]) * value
            flag += 1
            #score2 = math.log(len(self.base), 2) * len(self.base)
            end_score = score1+ end_score+len(key)
        reward = start_score - end_score

        if self.step == 15 or len(self.find_changeable(self.base))/len(self.base)<=0.2:
            self.change_base()
            done = True
            query1(self.result, self.index)
            query2(self.result, self.index)
            query3(self.result, self.index)
        else:
            done = False

        #r = self.find_changeable(self.base)
        #end_score = (1-len(r)/len(self.base))*100
        #reward= end_score-start_score
        #print(reward)
        return self.base, reward, done






