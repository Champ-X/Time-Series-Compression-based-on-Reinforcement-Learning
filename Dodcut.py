import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from  matplotlib import axes
def Dod(file):
    ts = []
    value = []
    with open(file, 'r', encoding='utf-8') as f:
        f.readline()
        temp = f.readlines()
        for i in range(6000):
            line=temp[i+8000]
            line = line.strip()
            before, _, after = line.partition(" ")
            ts.append(before)
            value.append(after)
    d = mytimestamp(ts)
    draw(value,d)





def draw(value,d):
    x1 = range(len(value))
    y1=[]
    for i in value:
        y1.append(float(i))
    x2 = d
    y2 = []
    for i in d:
        y2.append(float(value[i]))
    print(y2)
    plt.title("InlineSkate Part of data")  # 设置标题
    y = MultipleLocator(10000000)    # y轴每15一个刻度
    plt.xlabel(" ")  # 设置x坐标标签
    plt.ylabel("value")  # 设置y坐标标签
    plt.plot(x1, y1)
    #plt.scatter(x2, y2, color='red',marker='x')
    ax = plt.gca()
    ax.yaxis.set_major_locator(y)
    plt.show()

def mytimestamp(ts):
    print(len(ts))
    data =[]
    header =[]
    data.append(str(ts[0]))
    data.append(str(ts[1]))
    for i in range(2,len(ts)):
        d = (int(ts[i])-int(ts[i-1]))-(int(ts[i-1])-int(ts[i-2]))
        if d!=0:
            header.append(i)
    print(header)
    return header

Dod('data/InlineSkate')
