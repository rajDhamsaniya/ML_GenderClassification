import os

def rename_files():
	
	file_list = os.listdir("./yalefaces_2")
	print(file_list)
	#os.chdir("./male/training_set")
	i=0
	f_name=""
	for file_name in file_list : 
		#print("old name : " + file_name)
		#print("new_name : " + file_name.translate(None,"0123456789"))
		f_name = os.path.abspath(file_name)+".jpg"
		os.rename(os.path.abspath(file_name) , f_name)
		#print(file_name.split('.')[0])
		i = i+1
rename_files()