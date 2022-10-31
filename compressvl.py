import re
import sys
import time
import snappy
from sklearn.model_selection import train_test_split
from gensim.models.word2vec import Word2Vec
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten,Embedding,Dropout,LSTM,Bidirectional
from keras.optimizers import rmsprop_v2
from keras.models import load_model
from collections import deque
from matplotlib import pyplot
class RunLenghtEncoding():
    """ Suply methods for compressing / uncompressing text using RLE-Algorithm """

    regex_zero = '^0+'
    regex_one = '^1+'

    def encode(self, text):
        """ Encode text using RLE-Algorithm """
        if (len(text.strip()) == 0):
            return ''
        # Encode text to binary
        binary = ''.join(['%08d' % int(bin(ord(s))[2:]) for s in text])
        bin_lenght = len(binary)
        output = ''
        while (len(binary) > 0):
            zero = re.match(self.regex_zero, binary)
            one = re.match(self.regex_one, binary)
            if (zero):
                count = zero.group(0).count('0')
                if (count > 2):
                    output += str(count) + '0'
                else:
                    output += zero.group(0)
                binary = re.sub(self.regex_zero, '', binary, 1)
            elif (one):
                count = one.group(0).count('1')
                if (count > 2):
                    output += str(count) + '1'
                else:
                    output += one.group(0)
                binary = re.sub(self.regex_one, '', binary, 1)

        self.ratio = str(len(output) / bin_lenght)
        print(self.ratio)
        return output

    def decode(self, text):
        """ Decode text using RLE-Algorithm """
        output = ''
        while (len(text) > 0):
            if (text[0] == '1' or text[0] == '0'):
                output += text[0]
                text = text[1:]
            else:
                # Is compressed, need to decode
                output += str(text[1]) * int(text[0])
                text = text[2:]
        # Re-encode to ASCII
        return ''.join([chr(int(output[i * 8:i * 8 + 8], 2)) for i in range(0, int(len(output)) // 8)])
def zigzag(ts):
    if ts>=0:
        temp = ts*2
    else:
        temp = (0-ts)*2-1
    return temp

def rle(ts):
    start=ts[0]
    result = []
    result.append(ts[0])
    flag = 0
    for i in range(len(ts)-2):
        if ts[i+1]==start:
            flag=flag+1
        else:
            if flag!=0:
                result.append(flag)
                result.append(start)
                start = ts[i+2]
                flag=0
    return result
def ten2six(ts):
    ts = hex(int(ts))[2:]
    if len(ts) != 16:
        for k in range(16 - len(ts)):
            ts = '0' + ts
    return ts
def bitmask(ts):
    front = 0
    back = 0
    if ts[0]!='0' or ts[1]!='0':
        front = 0
    else:
        for i in range(1,7):
            if ts[2*i] != '0' or ts[2*i+1] != '0':
                front = i
                break
    if ts[14]!='0'or ts[15]!='0':
        back = 0
    else:
        for i in range(1,7):
            if ts[15-2*i] != '0' or ts[15-2*i-1] != '0':
                back = i-1
                break

    head1 = '{:03b}'.format(front)
    head2 = '{:03b}'.format(back)
    #print(head1+head2)
    rest=''
    for i in range(16-2*front-2*back):
        print(ts[2*front+i])
        rest=rest+'{:04b}'.format(int(ts[2*front+i],16))
    #print(head1+head2+rest)
    return '1110'+head1+head2+rest
def zero(ts):
    head = ''
    data = ''
    r = ''
    for i in range(8):
        if ts[2*i] ==0 and ts[2*i+1]==0:
            head=head+'0'
        else:
            head=head+'1'
            data =data+ts[2*i]+ts[2*i+1]
    for d in data:
        r=r+'{:04b}'.format(int(d,16))
    return '1111'+head+r

def isNumber(x):
    try:
        if x==int(x):
            return True
    except ValueError:
        return False

def methodA(ts):
    result =[]
    data = '0'+ten2six(ts[0])+ten2six(ts[1])
    for i in range(2,len(ts)):
        d = (ts[i] - ts[i - 1]) - (ts[i - 1] - ts[i - 2])
        result.append(d)
    temp = snappy.compress(str.encode(str(result)))
    l = sys.getsizeof(data)+sys.getsizeof(temp)
    return l

def methodB(ts):
    result =''
    data = '10'+ten2six(ts[0])+ten2six(ts[1])
    for i in range(2,len(ts)):
        d = (ts[i] - ts[i - 1]) - (ts[i - 1] - ts[i - 2])
        if isNumber(d) ==True:
            result=result+str(zigzag(int(d)))
        else:
            return False
    l = sys.getsizeof(data)+sys.getsizeof(result)
    return l

def methodC(ts):
    result =[]
    data = '110'+ten2six(ts[0])+ten2six(ts[1])
    for i in range(2,len(ts)):
        d = (ts[i] - ts[i - 1]) - (ts[i - 1] - ts[i - 2])
        result.append(d)
    temp = rle(result)
    l = sys.getsizeof(data)+sys.getsizeof(temp)
    return l

def methodD(ts):
    data = []
    data.append(ten2six(ts[0]))
    data.append(ten2six(ts[1]))
    for i in range(2,len(ts)):
        d = int((ts[i] - ts[i - 1]) - (ts[i - 1] - ts[i - 2]))
        if isNumber(d)!=True:
            return False
        if d == 0:
            # header.append('0')
            data.append('0')
        else:
            # header.append('1')
            if d in range(-4, 4):
                if d < 0:
                    d = '{:02b}'.format(-d)
                    data.append('101' + d)
                else:
                    d = '{:02b}'.format(d)
                    data.append('100' + d)
            elif d in range(-16, 16):
                if d < 0:
                    d = '{:04b}'.format(-d)
                    data.append('1101' + d)
                else:
                    d = '{:04b}'.format(d)
                    data.append('1100' + d)
            else:
                s = ten2six(ts[i])
                if len(s) != 16:
                    for k in range(16 - len(s)):
                        s = '0' + s
                data.append(bitmask(s))
    l = sys.getsizeof(data)
    return l

def methodE(ts):
    data = []
    data.append(ten2six(ts[0]))
    data.append(ten2six(ts[1]))
    for i in range(2,len(ts)):
        d = int((ts[i] - ts[i - 1]) - (ts[i - 1] - ts[i - 2]))
        if d<0:
            d = '1'+str(bin(d)[3:])
        else:
            d=str(bin(d)[2:])
        if set(d)!= {'0'}:
            b = d.strip('0')
        else:
            b='0'
        data.append(b)
    l = sys.getsizeof(data)
    return l

def methodF(ts):
    result =''
    data = '110'+ten2six(ts[0])+ten2six(ts[1])
    for i in range(2,len(ts)):
        d0 = ts[i] - ts[i - 1]
        d1 = ts[i - 1] - ts[i - 2]
        temp = d0^d1
        result = result+bin(temp)[2:]
    l = sys.getsizeof(data)+sys.getsizeof(result)
    return l

def choice(ts):
    c =[]
    c.append(methodA(ts))
    c.append(methodB(ts))
    c.append(methodC(ts))
    c.append(methodD(ts))
    c.append(methodE(ts))
    c.append(methodF(ts))
    #print(sys.getsizeof(ts))
    t=c[0]
    flag = 0
    for i in range(6):
        if c[i]==False:
            continue
        if c[i]<=t:
            t=c[i]
            flag =i

    return flag,t

def getcsv():
    value = []
    with open('data/UWaveGestureLibraryAll','r',encoding='utf-8') as f:
        f.readline()
        for line in f.readlines():
            line=line.strip()
            before,_,end =line.partition(" ")
            value.append(end)
    orig = sys.getsizeof(value)
    temp = 0
    v=2000
    result = []
    label =[]
    with open('train0.csv','w',encoding='utf-8') as file:
        #file.write("value,label"+'\n')
        data = []
        for i in range(int(len(value)/v)):
            for j in range(v):
                file.write(value[v*i+j]+' ')
                data.append(int(float(value[v*i+j])))
            #r,n = choice(data)
            n=methodA(data)
            r =0
            result.append(str(data))
            file.write(","+str(r)+'\n')
            label.append(str(r))
            temp =temp+n
            data =[]
    with open('traindata0.csv', 'w', encoding='utf-8') as file:
        for i in range(len(result)):
            file.write(result[i].strip(']').strip('[')+'\n')
    with open('trainlabel0.csv', 'w', encoding='utf-8') as file:
        for i in range(len(label)):
            file.write(label[i]+'\n')
    print(float(orig)/float(temp))
    return result,label

def build_vector(text, size, wv):
    # 创建一个指定大小的数据空间
    vec = np.zeros(size).reshape((1, size))
    # count是统计有多少词向量
    count = 0
    # 循环所有的词向量进行求和
    for w in text:
        try:
            vec += wv[w].reshape((1, size))
            count += 1
        except:
            continue
    # 循环完成后求均值
    if count != 0:
        vec /= count
    return vec


def wordv(x_train,x_test):
    print(len(x_train))
    wv = Word2Vec(size=300, min_count=5)
    wv.build_vocab(x_train)
    # 训练并建模
    wv.train(x_train, total_examples=1, epochs=1)
    # 获取train_vecs
    word_vec = []
    for i in wv.wv.index2word:
        word_vec.append(wv.wv[i])
    print(len(word_vec))
    print(word_vec)
    train_vecs = np.concatenate([build_vector(z, 300, wv) for z in x_train])
    # 保存处理后的词向量
    #np.save('train_vecs.npy', train_vecs)
    # 保存模型
    wv.save("modelword.pkl")
    wv.train(x_test, total_examples=1, epochs=1)
    test_vecs = np.concatenate([build_vector(z, 300, wv) for z in x_test])
    #np.save('data/test_vecs.npy', test_vecs)
    return train_vecs,test_vecs

def prepare():
    label = []
    with open('traindata.csv', 'r', encoding='utf-8') as f:
        data = f.readlines()
    x =np.array(data)
    with open('trainlabel.csv', 'r', encoding='utf-8') as f:
        t = f.readlines()
        for k in t:
            label.append(int(k))
    y = np.array(label)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)
    train_vecs,test_vecs = wordv(X_train,X_test)
    print(train_vecs)
    print(len(train_vecs))
    print(y_train)
    return train_vecs,y_train,test_vecs,y_test

def getmodel():
    voc_size =1000
    embedding_vector_features =10
    sent_length =300
    model = Sequential()
    model.add(Embedding(voc_size, embedding_vector_features, input_length=sent_length))
    model.add(Bidirectional(LSTM(64,activation="relu")))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    X_train, y_train,X_test, y_test,=prepare()
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=10)
    model.save('lstm.h5')
    pyplot.plot(history.history['loss'])
    # pyplot.plot(history.history['val_loss'])
    pyplot.title('model train loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train'], loc='upper right')
    pyplot.show()

def run(file):
    value = []
    label = []
    with open('data/result'+file+'.csv', 'r', encoding='utf-8') as f:
        l =f.readlines()
    for i in l:
        label.append(i.strip('\n'))
    with open('data/'+file, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f.readlines():
            line = line.strip()
            before, _, end = line.partition(" ")
            value.append(end)
    orig = sys.getsizeof(value)
    temp = 0
    v = 1500
    result = []
    data = []
    for i in range(int(len(value) / v)):
        for j in range(v):
            data.append(int(float(value[v * i + j])))
        result.append(data)
        data = []
    start = time.process_time()
    for i in range(len(result)):
        if label[i]=='0':
            temp=temp+methodA(result[i])
        elif label[i]=='1':
            temp=temp+methodB(result[i])
        elif label[i]=='2':
            temp=temp+methodC(result[i])
        elif label[i]=='3':
            temp=temp+methodD(result[i])
        elif label[i]=='4':
            temp=temp+methodE(result[i])
        elif label[i]=='5':
            temp=temp+methodF(result[i])
    print(time.process_time() - start)
    print((float(orig)/float(temp)))

#getmodel()
getcsv()
"""
file =['Server62','Server77','Server82','Server94','Server115','CinC_ECG_torso','MALLAT','Phoneme']
for i in range(3):
    run(file[i])
"""