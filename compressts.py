import os
import sys
import time

def ten2six(ts):
    ts = hex(int(ts))[2:]
    if len(ts) != 16:
        for k in range(16 - len(ts)):
            ts = '0' + ts
    return ts

def bitmask(ts):
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

def pk(ts):
    t1 = bitmask(ts)
    t2 = zero(ts)
    if len(t1) > len(t2):
        return t2
    else:
        return t1

def mytimestamp(ts):
    data =[]
    #header =[]

    data.append(ten2six(ts[0]))
    data.append(ten2six(ts[1]))
    for i in range(2,len(ts)):
        d = (ts[i]-ts[i-1])-(ts[i-1]-ts[i-2])
        if d==0:
            #header.append('0')
            data.append('0')
        else:
            #header.append('1')
            if d in range(-4,4):
                if d<0:
                    d= '{:02b}'.format(-d)
                    data.append('101'+d)
                else:
                    d = '{:02b}'.format(d)
                    data.append('100' + d)
            elif d in range(-16,16):
                if d<0:
                    d = '{:04b}'.format(-d)
                    data.append('1101' + d)
                else:
                    d = '{:04b}'.format(d)
                    data.append('1100' + d)
            else:
                d =pk(ten2six(ts[i]))
                data.append(d)

    return data


def readts(file):
    data = []
    old = 0
    new =0
    start = time.process_time()
    with open(file,'r',encoding='utf-8') as f:
        f.readline()
        lines = f.readlines()
        for line in lines:
            before, _, after = line.partition(" ")
            ts = hex(int(before))[2:]
            if len(ts) != 16:
                for k in range(16 - len(ts)):
                    ts = '0' + ts
            data.append(int(before))
            temp = int('0x' + ts,16)
            old = old+len(bin(temp)[2:])
    result = mytimestamp(data)
    r=''
    print(time.process_time() - start)
    for item in result:
        new = new +len(item)
        r=r+item
    #decode(r)
    ratio = float(old)/float(new)
    print(ratio)

def decode(ts):
    leng = len(ts)
    data=[]
    ts1 =ts[:16]
    ts2 =ts[16:32]
    data.append(int(ts1,16))
    data.append(int(ts2,16))
    mid= int(ts2,16)-int(ts1,16)
    rest = ts[32:]
    i=2
    flag=2
    d_start = time.process_time()
    while ts[32+i-2:]!='':
        if ts[30+i]=='0':
            data.append((data[flag-1]-data[flag-2])+data[flag-1])
            i=i+1
            flag=flag+1
        elif ts[30+i]=='1' and ts[30+i+1]=='0':
            temp = int(ts[30 + i + 3:30 + i + 5], 2)
            if ts[30+i+2]=='0':
                data.append((data[flag-1]-data[flag-2])+data[flag-1]+temp)
            else:
                data.append((data[flag - 1] - data[flag - 2]) + data[flag - 1] - temp)
            i=i+5
            flag = flag + 1
        elif ts[30+i]=='1' and ts[30+i+1]=='1' and ts[30+i+2]=='0':
            temp = int(ts[30 + i + 4:30 + i + 8], 2)
            if ts[30 + i + 3] == '0':
                data.append((data[flag - 1] - data[flag - 2]) + data[flag - 1] + temp)
            else:
                data.append((data[flag - 1] - data[flag - 2]) + data[flag - 1] - temp)
            i=i+8
            flag = flag + 1
        elif ts[30 + i] == '1' and ts[30 + i + 1] == '1' and ts[30 + i + 2] == '1' and ts[30 + i + 3] == '0':
            temp = ts[30 + i + 4:30 + i + 10]
            front = int(temp[:3],2)*2
            back = int(temp[3:],2)*2
            l =16-front-back
            bb =''
            for j in range(l):
                bb = bb+hex(int(ts[30 + i + 10+j*4:30 + i + 10 + j*4+4],2))[2:]
            for i in range(back):
                bb = bb+'00'
            data.append(int(bb,16))
            i=i+10+l*4
            flag=flag+1
        elif ts[30 + i] == '1' and ts[30 + i + 1] == '1' and ts[30 + i + 2] == '1' and ts[30 + i + 3] == '1':
            temp = ts[30 + i + 4:30 + i + 12]
            dd=''
            l=0
            for j in temp:
                if j=='0':
                    dd =dd+'00'
                else:
                    l=l+1
                    dd = dd +hex(int(ts[30+i+12+j*8:30+i+12+j*8+4],2))[2:]+hex(int(ts[30+i+12+j*8+4:30+i+12+j*8+8],2))[2:]
            data.append(int(dd,16))
            flag=flag+1
            i = i+12+l*8
    print(time.process_time() - d_start)



for i in range(8):
    readts(r'C:\Users\nanjiang\Desktop\ATimeSeriesDataset-master\IoT\IoT'+str(i))


#decode('000000005ac8497c000000005ac8497e000000101011001010101000010101100010000010101100010001010110001000101011000100010101100010101011000100010101100010001010110001000101011000100001010110001100011010110101100010000101011000100010101100101010110101100010000010101100010001010110001000000')


