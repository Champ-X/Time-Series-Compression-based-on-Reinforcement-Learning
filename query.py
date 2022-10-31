import re
import time
import gzip
import matplotlib.pyplot as plt
import numpy as np
"""
query1:
select * from ts 
where d_id = xxx 
and time > start_time 
and time <= end_time;
"""
def query1(result,index):
    t1 = time.process_time()
    start_time =100
    end_time =100000
    r ={}
    temp = 0
    i = 0
    value = list(index.values())
    key = list(index.keys())
    while temp<end_time-start_time:
        for item in value[i]:
            if item<=end_time and item>start_time:
                r[item]=key[i]
                if '#' in key[i]:
                    place = [m.start() for m in re.finditer("#", key[i])]
                    r[item] = list(r[item])
                    for j in range(len(place)):
                        #print(r[item][place[j]])
                        r[item][place[j]] = list(result[item].values())[0][j]
                    r[item] = ''.join(r[item])
                r[item] = int(r[item],2)
                temp = temp + 1
        i=i+1
    print(len(r))
    print('query1:')
    print(time.process_time()-t1)
    return r

"""
query2:
select * from ts
where d_id = xxx
where time > start_time
and time <= end_time
and value>1000;
"""
def query2(result,index):
    t1 = time.process_time()
    start_time =100
    end_time =100000
    v =1000
    r ={}
    rr = {}
    temp = 0
    i = 0
    value = list(index.values())
    key = list(index.keys())
    while temp<end_time-start_time:
        for item in value[i]:
            if item<=end_time and item>start_time:
                r[item]=key[i]
                if '#' in key[i]:
                    place = [m.start() for m in re.finditer("#", key[i])]
                    r[item]=list(r[item])
                    for j in range(len(place)):
                        r[item][place[j]]= list(result[item].values())[0][j]
                    r[item] = ''.join(r[item])
                r[item] = int(r[item],2)
                if r[item]<v:
                    rr[item]=r[item]
                temp = temp + 1
        i=i+1
    #print(rr)
    print('query2:')
    print(time.process_time()-t1)
    return rr

"""
query3:
select time from ts
where d_id = xxx
and value>700
and value<=1000;
"""
def query3(result,index):
    t1 = time.process_time()
    v_up =900
    v_down =1000
    r={}
    t=[]
    value = list(index.values())
    key = list(index.keys())
    for i in range(len(key)):
        if '#' in key[i]:
            temp = key[i]
            place = [m.start() for m in re.finditer("#", key[i])]
            if int(temp.replace('#', '1'), 2) <= v_down and int(temp.replace('#', '0'), 2) > v_up:
                for item in value[i]:
                    temp = list(temp)
                    for j in range(len(place)):
                        temp[place[j]] = list(result[item].values())[0][j]
                    r[item] = ''.join(temp)
                    if int(r[item], 2) <= v_down and int(r[item], 2) > v_up:
                        t.append(int(r[item], 2))
        else:
            if int(key[i], 2) <= v_down and int(key[i], 2) > v_up:
                for item in value[i]:
                    t.append(item)
    #print(len(rr))
    print('query3:')
    print(time.process_time()-t1)
    return t

"""
query1:
select * from ts 
where d_id = xxx 
and time > start_time 
and time <= end_time;
"""
def dequery1(ts):
    start_time = 100
    end_time = 100000
    #print(ts[start_time:end_time])
    return ts[start_time:end_time]

"""
query2:
select * from ts
where d_id = xxx
where time > start_time
and time <= end_time
and value>1000;
"""
def dequery2(ts):
    start_time = 100
    end_time = 100000
    v=1000
    r ={}
    for i in range(start_time,end_time):
        if ts[i]<=v:
            r[i]=ts[i]
    #print(len(r))
    return r

"""
query3:
select time from ts
where d_id = xxx
and value>700
and value<=1000;
"""
def dequery3(ts):
    v_up = 900
    v_down = 1000
    r = {}
    for i in range(len(ts)):
        if ts[i]<=v_down and ts[i]>v_up:
            r[i]=ts[i]
    #print(len(r))
    return r

def exp(file):
    ts =[]
    tt=[]
    with open(file,'r',encoding='utf-8') as f:
        lines = f.readlines()
    for item in lines:
        ts.append(int(item.strip('\n')))
    gz = gzip.compress(str.encode(str(ts)))
    t1 = time.process_time()
    temp = str(gzip.decompress(gz)).strip("'").strip('[').strip(']')
    t =temp.split(", ")
    for i in t:
        i =i.replace("b'[","")
        tt.append(int(i))
    dequery1(tt)
    print(time.process_time()-t1)
    t1 = time.process_time()
    temp = str(gzip.decompress(gz)).strip("'").strip('[').strip(']')
    t = temp.split(", ")
    for i in t:
        i = i.replace("b'[", "")
        tt.append(int(i))
    dequery2(tt)
    print(time.process_time() - t1)
    t1 = time.process_time()
    temp = str(gzip.decompress(gz)).strip("'").strip('[').strip(']')
    t = temp.split(", ")
    for i in t:
        i = i.replace("b'[", "")
        tt.append(int(i))
    dequery3(tt)
    print(time.process_time() - t1)

def drawpic():
    query =['Q1','Q2','Q3','Q4','Q5','Q6']
    l = np.arange(len(query))
    result1 = [0.171875,0.171875,0.171875,0.1875,0.40625,0.421875]
    result2 = [0.03125,0.0625,0.046875,0.0625,0.078125,0.046875]
    #result = date.groupby(date.index.year).agg(high=('最高价', 'mean'), low=('最低价', 'mean'))  # 分别计算每年股票最高价、最低价均值
    plt.bar(l, result1, width=0.2, color='lightskyblue', label='Gzip')  # 绘制每年股票最高价均值的条形图，颜色设置为红色
    plt.bar(l + 0.2, result2, width=0.2, color='yellowgreen', label='my')  # 绘制每年股票最低价均值的条形图，颜色设置为蓝色
    plt.xticks(l + 0.1, query)  # 让横坐标轴刻度显示时间，result.index+0.2为横坐标轴刻度的位置
    plt.ylim(0, 0.5)  # 设置y轴的显示范围
    plt.title('Query time')  # 设置标题
    plt.legend()
    plt.show()



#exp('data/mit101.txt')
drawpic()