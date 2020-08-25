#Script for the Deduplication of Records

import numpy as np
import csv
import recordlinkage as rl 
import pandas as pd

#Reading the CSV file
f1 = pd.read_csv(open('Deduplication Problem - Sample Dataset.csv','rb'))


data1 = pd.DataFrame(f1)
data1 = pd.DataFrame(data=data1,dtype='U')

#Making Record Pairs For Similar Data in Various Fields
block_class = rl.BlockIndex(on=["ln","dob","gn","fn"])
block_pairs = block_class.index(data1)
	
#Comparing each record pair with the data.

compare_class = rl.Compare()

compare_class.string('ln','ln',method = "jarowinkler",threshold = 0.85)
compare_class.exact('dob','dob')
compare_class.exact('gn','gn')
compare_class.string('fn','fn',method = "jarowinkler",threshold = 0.85)

compare_result = compare_class.compute(block_pairs,data1)  

#Filtering the Records further by comparing the Columns matched

matches = compare_result[compare_result.sum(axis=1) >= 3]

#Collecting the Indexes of the Data 

index_list = []
output_list=[]
for i in range(0,len(matches)):
	m,n = matches.index[i]
	output_list.append(m)
	index_list.append(n)

output_list=set(output_list)
index_list = set(index_list)	

for j in index_list:
	if j in output_list:
		output_list.remove(j)

output_list = set(output_list)


#wrtitng the Unique Record in the Output.csv file
write_data = []
for k in output_list:
	write_data.append(data1.ix[k])

write_data = pd.DataFrame(write_data)










































































write_data.to_csv('output.csv',index=False)
