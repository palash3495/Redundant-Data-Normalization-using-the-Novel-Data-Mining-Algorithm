#!/usr/bin/python
# coding: utf-8 -*-

import codecs
from collections import Counter
import pandas as pd
import time

import graph

start_time = time.time()
def find_duplicate_1(filename):

#empty set
    myset = set()
#empty list
    mylist = []

#read records.txt file 
    filename = codecs.open(filename,encoding='utf-8')

    log_file = codecs.open("log.txt",'w',encoding='utf-8')
    log_file.write("Below values are duplicate" + '\n' + '\n')

#Iterate through the file
#Normalize the feature
    for rec in filename:
        rec = rec.replace('\n',"")
        if rec != "":
           myset.add(rec)
           mylist.append(rec)

    #convert set into list
    #all duplicates removed
    newlist = list(myset)

    #Start selecting features
    c1 = Counter(mylist)
    c2 = Counter(newlist)

    diff = c1 - c2
    #Print duplicate values
    print(list(diff.elements()))
    newfile=pd.DataFrame(newlist)
    newfile.to_csv('denormalization.csv')





'''Change sample.txt with your file
call find_duplicate function'''

#find_duplicate_1("sample.txt")
find_duplicate_1("Datasets/ConferenceName_non_standard-1.csv")


end_time = time.time()
GFS_time=end_time-start_time
print("Total exection time:",GFS_time)

graph.accuracy()
graph.precision()
graph.recall()
graph.time_GFS(GFS_time)


