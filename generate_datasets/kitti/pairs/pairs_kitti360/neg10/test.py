import numpy as np
import sys
import os
def conv(seq='0000'):
    seq_temp=seq[-2:]
    pairs=np.genfromtxt(seq+'.txt',dtype='int')
    with open(seq_temp+'.txt','w') as f:
        for pair in pairs:
            pair=[os.path.join(seq_temp,str(v).zfill(10)+'.json') for v in pair]
            f.write(pair[0]+" "+pair[1]+"\n")

if __name__=="__main__":
    seq='0000'
    if len(sys.argv)>1:
        seq=sys.argv[1]
    conv(seq)
