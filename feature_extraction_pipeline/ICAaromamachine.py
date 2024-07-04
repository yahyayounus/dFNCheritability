import shutil
import os

#list of subjects inputted into the preprocessing pipeline
with open('/mnt/d/listdelete.txt', 'r') as f:
	list = [word for line in f for word in line.split()]

number_of_words = 0
 
with open(r'/mnt/d/listdelete.txt','r') as file2:
 
    data = file2.read()

    lines = data.split()
 
    number_of_words += len(lines)


#automates ICA-aroma for each subject 
#for information regarding how to run ICA_AROMA, please visit the following repo: https://github.com/maartenmennes/ICA-AROMA
for x in range(number_of_words):
	l="python2.7 ~/ICA-AROMA-master/ICA_AROMA.py -feat /mnt/d/dFNCheritability/{}/FEAT_results.feat -out /mnt/d/dFNCheritability/{}/ICA_AROMA_results".format(list[x], list[x])
	print(l)
