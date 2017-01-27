#Our task is to make the commmunities data usabel,
#i.e. replace the '?' with the mean of the samples of
# the corresponding feature


#program takes in "communities.data and outputs communities_usable.data"
#after replacing '?' with means in the corresponding columns.

#replacing by the mean of the data might not be the best idea
#if the features have correlations between them. If a feature is 
#highly correlatedwith the other features, then the sample might be far off
#from the sampe mean and is strongly dependant on the other data

import string

#start two iterators
fdr1 = open("communities.data", 'r')
fdr2 = open("communities.data", 'r')

#initialize variables
mean = 0
count = 0
line_count = 0
feat_count = 0

#read CSV and compute mean of each column
for line in fdr1:
	words = line.split(',')
	line_count = line_count+1

	if count == 0:
		feat_count = len(words)
		count = [0]*len(line)
		mean = [0]*len(line)
	for i in range(feat_count):
		if i == 3:
			d = 4
		elif(words[i] == '?'):
			count[i] = count[i]+1
		else:
			mean[i] = mean[i] + float(words[i])


#close first iterator
fdr1.close()

#the data is written into this file
fdw = open("communities_usable.data", "w")

#Everytine==me a ? is encountered, it is replaced by the mean
# the output file is written line-by-line
for line in fdr2:
	words = line.split(',')
	to_write = ""
	for i in range(feat_count):
		if words[i] == '?':
			to_write = to_write + str(mean[i]/(line_count-count[i]))
		else:
			to_write = to_write + words[i]
		if i != feat_count-1:
			to_write = to_write + ","
	fdw.write(to_write)


#close second iterator, output file descriptor
fdr2.close()
fdw.close()

