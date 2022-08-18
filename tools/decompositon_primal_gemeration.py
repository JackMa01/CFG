import json
import argparse
parser = argparse.ArgumentParser()
parser.add_argument( '-JDC', '--java_decomposition_file', type=str, default='', help='Path of java decomposition file')
args = parser.parse_args()

ChS = ['⿰', '⿱', '⿲', '⿳', '⿴', '⿵', '⿶', '⿷', '⿸', '⿹', '⿺', '⿻'] #Character Struct Sign
DC = {} #decomposition dictionary
PR = [] #primals list

with open(args.java_decomposition_file, 'r', encoding='utf8') as fp:
    JDC = fp.readlines()

for dc in JDC:
    dc = dc[:-1]
    dc = dc.split('\t')
    value = []
    if dc[0] in ChS:
        continue
    for char in dc[2]:
        if char not in ChS:
            value.append(char)
            PR.append(char)
    DC[dc[0]] = value

PR = list(set(PR))

with open('decomposition.json', 'w', encoding='utf8') as fp:
    json.dump(DC, fp)
with open('primals.json', 'w', encoding='utf8') as fp:
    json.dump(PR, fp)
