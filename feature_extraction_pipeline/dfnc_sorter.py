import shutil
import os

#The dFNC_sorter.txt file is a text file that has the text outputted from GIFT
#It should look like this:
#            (make sure to have a empty line)
#val(:,:,1)=
#            (make sure to have a empty line)
#   6
#   4
#   2
#   etc
#            (make sure to have a empty line)
#            (make sure to have a empty line)
#val(:,:,2)=
#            (make sure to have a empty line)
#   6
#...
#This will be converted into a sequence of numbers for each subject.

#This file will be created in order for the program to run but has nothing
#to do with the output
q=  open('/mnt/d/dfnc_sorter_lines.txt', 'r+')


	
#Inputs the amount of subjects in order to calculate the lines
o = int(input("How many subjects: "))
#Inputs time points
trs = int(input("How many time points do you have (largest x in val(:,:,x)): "))

#Calculates each line for the subject
for p in range(o):

	l = p + 1

	

	k= "Subject{}lines:".format(l)
	for x in range(trs):

		r = x*(o) + 1*x + p + 1

		b = str(r) + "\n"

		q.write(b)

with open("/mnt/d/dfnc_sorter.txt", "r") as file:

	content = file.read().splitlines()

clean_content = []

for ele in content:
	if ele.strip():
		ele.replace(" ","")
		clean_content.append(ele)


with open("/mnt/d/dfnc_sorter_lines.txt", "r") as file2:
	#Ignore this file creation
	content2 = file2.read().splitlines()

gh= len(content2)


writestates=  open('/mnt/d/dfnc_sorter_states.txt', 'w')
#this is where your file will be outputted

z = range(0,gh)

lst= list(range(1,o))

for x in z:
	yu= int(content2[x])
	
	if yu in lst:
		print(yu-1)
		writestates.write("\n")
		writestates.write(str(yu -1))
		


	hj= clean_content[yu]
	writestates.write(str(hj))

print("\nComplete! The output is in the dfnc_sorter_states.txt file.")
