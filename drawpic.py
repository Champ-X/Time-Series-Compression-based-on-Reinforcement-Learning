import gorillacompression as gc
import sys
import snappy
import time
import zlib
import gzip
import bz2


def gorillaencodetimestamp():
    lines=[]
    timestamp=[]
    ratio=[]
    for i in range(8):
        with open('/home/nan/ATimeSeriesDataset/IoT/IoT'+str(i), 'r') as file:
            num = file.readline()
            for line in file.readlines():
                line = line.strip()
                lines.append(line)
                before, _, _ = line.partition(" ")
                timestamp.append(int(before))
        content = gc.TimestampsEncoder.encode_all(timestamp)
        # print(sys.getsizeof(content['encoded']))
        # print(sys.getsizeof((timestamp)))
        ratio.append(float(sys.getsizeof(str(timestamp)) / sys.getsizeof(str(content['encoded']))))
        #print(ratio)
        lines=[]
        timestamp=[]
    return ratio

def gorillaencode():
    lines = []
    timestamp = []
    value = []
    ratio = []
    for i in range(1):
        with open('/home/nan/ATimeSeriesDataset/Server/Server48', 'r') as file:
            num = file.readline()
            for line in file.readlines():
                line = line.strip()
                lines.append(line)
                before, _, after = line.partition(" ")
                timestamp.append(int(before[:-4]))
                value.append(float(after))
            pairs = list(zip(timestamp, value))
        content = gc.PairsEncoder.encode_all(pairs)
        #print(pairs)
        print(sys.getsizeof(content['encoded']))
        print(sys.getsizeof((timestamp)))
        ratio.append(float(sys.getsizeof(str(pairs)) / sys.getsizeof(str(content['encoded']))))
        # print(ratio)
        lines = []
        timestamp = []
    return ratio

#Gorilla, snappy, Zlib, bz2, gzip
def encodetimestamp(file):
    lines = []
    timestamp = []
    with open(file,'r') as f:
        num = f.readline()
        for line in f.readlines():
            line = line.strip()
            lines.append(line)
            before, _, _ = line.partition(" ")
            timestamp.append(int(before))

    return timestamp

def encodevalue(file):
    lines = []
    timestamp = []
    with open(file,'r') as f:
        #num = f.readline()
        for line in f.readlines():
            line = line.strip()
            lines.append(line)
            before, _, after = line.partition(" ")
            timestamp.append(float(line))

    return timestamp

if __name__ == '__main__':
    #gorillats=gorillaencodetimestamp()
    #print(gorillaencode())
    ratio = {'gorilla':[], 's':[],'z':[],'b':[],'gz':[],'influx':[],'my':[]}
    t = {'gorilla':[], 's':[],'z':[],'b':[],'gz':[],'influx':[],'my':[]}
    files=['data/mit100.txt','data/mit101.txt','data/mit102.txt','data/mit103.txt','data/mit104.txt','data/mit105.txt']
    for i in range(len(files)):
        timestamp = encodevalue(files[i])
        start = time.process_time()
        gorilla = gc.ValuesEncoder.encode_all(timestamp)
        #gc.TimestampsDecoder.decode_all(gorilla)
        t['gorilla'].append(time.process_time()  - start)
        ratio['gorilla'].append(float(sys.getsizeof(str(timestamp)) / sys.getsizeof(str(gorilla['encoded']))))
        start = time.process_time()
        s = snappy.compress(str.encode(str(timestamp)))
        #snappy.decompress(s)
        t['s'].append(time.process_time()  - start)
        ratio['s'].append(float(sys.getsizeof(str(timestamp)) / sys.getsizeof(str(s))))
        start = time.process_time()
        z = zlib.compress(str.encode(str(timestamp)))
        #zlib.decompress(z)
        t['z'].append(time.process_time()  - start)
        ratio['z'].append(float(sys.getsizeof(str(timestamp)) / sys.getsizeof(str(z))))
        start = time.process_time()
        b = bz2.compress(str.encode(str(timestamp)))
        #bz2.decompress(b)
        t['b'].append(time.process_time() - start)
        ratio['b'].append(float(sys.getsizeof(str(timestamp)) / sys.getsizeof(str(b))))
        start = time.process_time()
        gz = gzip.compress(str.encode(str(timestamp)))
        #gzip.decompress(gz)
        t['gz'].append(time.process_time()  - start)
        ratio['gz'].append(float(sys.getsizeof(str(timestamp)) / sys.getsizeof(str(gz))))
    ratio['influx']=[2.4,3.7,3.23,3.31,2.78,3.09,2.81,2.43]
    ratio['my']=[3.41,5.56,5.10,8.14,8.01,5.18,4.61,3.92]
    t['influx']=[0.047,0.047,0.047,0.047,0.031,0.046,0.047,0.031]
    t['my']=[0.62,0.57,0.59,0.50,0.41,0.53,0.57,0.54]
    #t['gorilla']= [0.72, 0.75, 0.72, 0.51, 0.33, 0.59, 0.74, 0.68]
    #print(ratio)
    x = range(8)
    print(ratio)
    print(t)
    #drawline(ratio,x)
    #drawtime(ratio,x)
    #ts=[1523075452,1523075454,1523075456,1523075458,1523075460,1523075462,1523075464,1523075466,1523075467,1523075470,1523075472,1523075474,1523075476,1523075478,1523075480]
    #mytimestamp(ts)
