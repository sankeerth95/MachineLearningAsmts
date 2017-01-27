import os
import sys
import numpy as np
import utility as utl
#Copy the binary fies l1_logreg_classify, l1_logreg_train and l1_logred_regpath
#onto present directory, and then run this python script

lambda1 = 0.0
if len(sys.argv[1]) > 1:
	lambda1 = sys.argv[1]

os.system("./l1_logreg_train -s Train_features_prob9 Train_labels_prob9 "+str(lambda1)+" l1reg_prob9")
os.system("./l1_logreg_classify l1reg_prob9 Test_features_prob9 result_test_prob9")

#Gets data from file, parses them and puts it into 
#list  ylist, being feature and output respectively 
def get_predicted(labels_file):

	fdlabel = open(labels_file, "r")
	for line in fdlabel:
		if line[0] != '%':
			break
	
	ylist = [float(line.split()[-1]) for line in fdlabel]
	fdlabel.close()
	return ylist

def get_y_test(labels_file):

	fdlabel = open(labels_file, "r")
	
	ylist = [float(line.split(',')[-1]) for line in fdlabel]
	fdlabel.close()
	return ylist

ypredicted = np.array(get_predicted("result_test_prob9"))
ytest = np.array(get_y_test("DS2-test.csv"))
performance = utl.get_perf_params(ypredicted, ytest)
utl.write_data("output_prob9.txt", [], performance)
