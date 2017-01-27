import os


#open_files[i] containd file desctiptors of files in parti foder
#files[i] contains the filenames of files in parti folder

def get_data(split):


	vocabulary = set()
	X_train_list = []
	y_train_list = []
	X_test_list = []
	y_test_list = []

	for i in xrange(10):
		st = "part"+str(i+1)+"/"
		path = "./Q10/"+st
		files = [fdesc for fdesc in os.listdir(path) if fdesc.endswith(".txt")]
		open_files = [open(path+fl, "r") for fl in files]

		if i != split and i != (split+1):

			for fd_el in open_files:

				x = dict()
				build_feature_vector(x, fd_el)
				X_train_list.append(x)
				vocabulary=vocabulary.union(set(x.keys()))

			y_train_list.extend(output_vector(files))
	
		else:
			for fd_el in open_files:

				x = dict()
				build_feature_vector(x, fd_el)
				X_test_list.append(x)
	
			y_test_list.extend(output_vector(files))


		for file_instance in open_files:
			file_instance.close()

	return X_train_list, y_train_list, X_test_list, y_test_list, vocabulary

def output_vector(filenames):
	y = [0]*len(filenames)
	i = 0
	for filename in filenames:
		if filename.rfind("legit") != -1:
			y[i]=1
		elif filename.rfind("spmsg") != -1:
			y[i]=0
		i+=1
	return y

def build_feature_vector(x, f_instance):

	for line in f_instance:
		words = line.split()
		for word in words:
			x[word] = x.get(word, 0)+1
	if "Subject:" in x.keys():
		del x["Subject:"]


#for file_instance in fd_array:
#	file_instance.close()
