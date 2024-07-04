import numpy as np
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings("ignore")
import timeit
import datetime
from sklearn import *
from sklearn.model_selection import train_test_split
from sklearn import svm
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier
import subprocess
import random
from sklearn.svm import LinearSVC, SVC
from random import randint
from sklearn.linear_model import LinearRegression, SGDClassifier, LogisticRegression
from sklearn.neural_network import MLPClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoLocator
import pandas as pd
import seaborn as sns
import timeit

### represent a sequence in form of items and sequential patterns (SPs),
### learn sequence vectors using Doc2Vec (PV-DBOW) from items and SPs separately
### take average of two sequence vectors
### use SVM or appropiate model(s) as the classifier

### variables ###
data_name = "dFNC"
path = "./data/" + data_name
minSup = 0.45
gap = 4# 0: any gap or >0: use gap constraint
dim = 512 ##################### look at this #########################
n_run = 10 # 60 works best

### functions ###
# mine SPs from sequences
def mine_SPs(file_seq, minSup, gap, file_seq_sp, file_sp, file_seq_items_sp):
    subprocess.run("sp_miner.exe -dataset {} -minsup {} -gap {} -seqsp {} -sp {} -seqsymsp {}".
                   format(file_seq, minSup, gap, file_seq_sp, file_sp, file_seq_items_sp))

# load sequences in form of items and their labels
def load_seq_items(file_name):
    labels, sequences = [], []
    with open(file_name) as f:
        for line in f:
            label, content = line.split("\t")
            if content != "\n":
                labels.append(label)
                sequences.append(content.rstrip().split(" "))
                
    return sequences, labels
    print("sequences")
    print(sequences)

# load sequences in form of SPs and their labels
def load_seq_SPs(file_name):
    labels, sequences = [], []
    with open(file_name) as f:
        for line in f:
            label, content = line.split("\t")
            labels.append(label)
            sequences.append(content.rstrip().split(" "))
    return sequences, labels

# create a sequence id to each sequence
def assign_sequence_id(sequences):
    sequences_with_ids = []
    for idx, val in enumerate(sequences):
        sequence_id = "s_{}".format(idx)
        sequences_with_ids.append(TaggedDocument(val, [sequence_id]))
    return sequences_with_ids

start_date_time = datetime.datetime.now()
start_time = timeit.default_timer()

#LOADS IN DATA
print("\n")
print("### sqn2vec_sep_classify, data: {}, minSup={}, gap={}, dim={} ###".format(data_name, minSup, gap, dim))
# mine SPs and associate each sequence with a set of SPs
in_seq = path + "/{}.txt".format(data_name)
out_seq_items_sp = path + "/{}_seq_items_sp_{}_{}.txt".format(data_name, minSup, gap)
out_seq_sp = path + "/{}_seq_sp_{}_{}.txt".format(data_name, minSup, gap)
out_sp = path + "/{}_sp_{}_{}.txt".format(data_name, minSup, gap)
mine_SPs(in_seq, minSup, gap, out_seq_sp, out_sp, out_seq_items_sp)
# load sequences in the form of items
data_path = path + "/" + data_name + ".txt"
print("\n")
data_i_X, data_i_y = load_seq_items(data_path)
print("\n")
# assign a sequence id to each sequence
data_seq_i = assign_sequence_id(data_i_X)
# load data in the form of patterns
data_path = path + "/{}_seq_sp_{}_{}.txt".format(data_name, minSup, gap)
data_p_X, data_p_y = load_seq_SPs(data_path)
# assign a sequence id to each sequence
data_seq_p = assign_sequence_id(data_p_X)

all_acc, all_mic, all_mac, all_aroc = [], [], [], []

accuracyvisual= []
arocvisual= []
runvisual= []
for run in range(n_run):
    print("\n")
    print("run={}".format(run))
    # learn sequence vectors using Doc2Vec (PV-DBOW) from items
    d2v_i = Doc2Vec(vector_size=dim, min_count=0, workers=16, dm=0, epochs=50)
    d2v_i.build_vocab(data_seq_i)
    d2v_i.train(data_seq_i, total_examples=d2v_i.corpus_count, epochs=d2v_i.epochs)
    data_i_vec = [d2v_i.docvecs[idx] for idx in range(len(data_seq_i))]
    del d2v_i  # delete unneeded model memory
    # learn sequence vectors using Doc2Vec (PV-DBOW) from SPs
    d2v_p = Doc2Vec(vector_size=dim, min_count=0, workers=16, dm=0, epochs=50)
    d2v_p.build_vocab(data_seq_p)
    d2v_p.train(data_seq_p, total_examples=d2v_p.corpus_count, epochs=d2v_p.epochs)
    data_p_vec = [d2v_p.docvecs[idx] for idx in range(len(data_seq_p))]
    del d2v_p  # delete unneeded model memory
    # take average of sequence vectors
    data_i_vec = np.array(data_i_vec).reshape(len(data_i_vec), dim)
    data_p_vec = np.array(data_p_vec).reshape(len(data_p_vec), dim)
    data_vec = (data_i_vec + data_p_vec) / 2

    # generate train and test vectors using 10-fold CV
    train_vec, test_vec, train_y, test_y = \
        train_test_split(data_vec, data_p_y, test_size=0.5, random_state=run, stratify=data_p_y)
    


    # INSTEAD OF svm.LinearSVC() from standard pipeline, the following was modified:
    ###################################################
    svm_d2v = OneVsRestClassifier(RandomForestClassifier())
    ###################################################
    # classify test data
    svm_d2v.fit(train_vec, train_y)
    test_pred = svm_d2v.predict(test_vec)

    acc = accuracy_score(test_y, test_pred)

    # Assuming 'test_y' is the true labels and 'test_pred' is the predicted probabilities
    test_pred2 = [float(pred.split('-')[-1]) for pred in test_pred]
    test_y2 = [float(test2.split('-')[-1]) for test2 in test_y]

    test_y2 = np.array(test_y2)
    test_pred2 = np.array(test_pred2)

    classes= np.unique(test_y2)
    num_classes = len(classes)

    class_auc_roc_values = []

    for class_idx in range(num_classes):
        # Compute AUC-ROC for each class
        auc_roc = roc_auc_score(test_y2 == classes[class_idx], test_pred2)
        class_auc_roc_values.append(auc_roc)
    print(class_auc_roc_values)
    aroc = np.average(class_auc_roc_values)

    #aroc = roc_auc_score(test_y2, test_pred2, multi_class='ovo')


    mic = f1_score(test_y, test_pred, pos_label=None, average="micro")
    mac = f1_score(test_y, test_pred, pos_label=None, average="macro")
    all_acc.append(acc)
    all_mic.append(mic)
    all_mac.append(mac)
    all_aroc.append(aroc)
    # obtain accuracy and F1-scores
    accuracyRounded=np.round(acc, 4)
    arocRounded=np.round(aroc, 4)
    print("accuracy: {}".format(accuracyRounded))
    print("AROC: {}".format(arocRounded))
    print("micro: {}".format(np.round(mic, 4)))
    print("macro: {}".format(np.round(mac, 4)))

    runadd= run + 1
    runvisual.append(runadd)
    accuracyadd= accuracyRounded * 100
    arocadd= arocRounded * 100
    accuracyvisual.append(accuracyadd)
    arocvisual.append(arocadd)

visuals= pd.DataFrame({"Run Number": runvisual, "Accuracy": accuracyvisual})
arocvisuals= pd.DataFrame({"Run Number": runvisual, "AROC": arocvisual})

print("\n")
print("avg accuracy: {} ({})".format(np.round(np.average(all_acc), 4), np.round(np.std(all_acc), 3)))
print("avg AROC: {} ({})".format(np.round(np.average(all_aroc), 4), np.round(np.std(all_aroc), 3)))
print("avg micro: {} ({})".format(np.round(np.average(all_mic), 4), np.round(np.std(all_mic), 3)))
print("avg macro: {} ({})".format(np.round(np.average(all_mac), 4), np.round(np.std(all_mac), 3)))

end_date_time = datetime.datetime.now()
end_time = timeit.default_timer()
print("\n")
print("start date time: {} and end date time: {}".format(start_date_time, end_date_time))
print("runtime: {}(s)".format(round(end_time-start_time, 2)))

#plotting results
sns.set_theme()
fig = sns.relplot(data=visuals, x="Run Number", y="Accuracy", kind="line", errorbar="sd")
plt.yticks([0, 5, 10, 15, 20])

plt.show()


fig2 = sns.relplot(data=arocvisuals, x="Run Number", y="AROC", kind="line", errorbar="sd")
fig2.ax.yaxis.set_major_locator(AutoLocator())

plt.show()



mode = 2


#opening text files
with open(r'C:\Users\dr_ku\Desktop\Sqn2Vec-master\data\dFNC\dFNC.txt', 'r') as f:
    wordlist = [line.split(None, 1)[0] for line in f]


with open(r'C:\Users\dr_ku\Desktop\Sqn2Vec-master\listofsubjects.txt', 'r') as f:
    wordlist2 = [line.split(None, 1)[0] for line in f]
    print(wordlist2)


try:
    while True:
        if mode == 2:
            print("\nMode 2 being used... Press A to exit")
            print("\n")
            humanid= "sub-"+input("What is the ID for Subject 1 (write number only including zeros):")
            print(humanid)
            

            if humanid in wordlist2:
                print("\n")
                print("ID Found!")
                humanidplace = wordlist2.index(humanid)
            else:
                print("\n")
                print("ID Not found...")


            familyid = wordlist[humanidplace]

            if familyid in data_p_y:
                print("\n")
                print("Family Found!")
                familyplace = data_p_y.index(familyid)
            else:
                print("\n")
                print("Family Not found...")




        prediction = svm_d2v.predict(data_vec[familyplace].reshape(1, -1)) 
        print("\n")
        print("Predicted Family ID:%s, Actual Family ID: %s" % (prediction, data_p_y[familyplace]))


        s = prediction
        matched_indexes = []
        i = 0
        length = len(wordlist)

        while i < length:
            if s == wordlist[i]:
                matched_indexes.append(i)
            i += 1


        final_twins= []

        for t in range(0, len(matched_indexes)):
            fds= matched_indexes[t]
            z = wordlist2[fds]
            final_twins.append(z)
        print("\n")
        print("The Twins for that Subject Are:")
        print(final_twins)
except KeyboardInterrupt:
    pass
