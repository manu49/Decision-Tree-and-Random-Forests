import numpy as np
import csv
import math

import sys


filename1 = sys.argv[1]
filename2 = sys.argv[3]
filename3 = sys.argv[2]

outputfile = sys.argv[4]

raw_train_data = []
raw_test_data = []
raw_val_data = []

raw_test_labels = []
raw_train_labels = []
raw_val_labels = []

training_data = []
testing_data = []
val_data = []

with open(filename1, 'r') as csvfile: 
    csvreader = csv.reader(csvfile)  
    fields = next(csvreader)  
    l = 0
    for row in csvreader: 
        if(l==0):
            l+=1
        else:
            k = len(row)
            raw_train_data.append(row[0:(k-1)])
            raw_train_labels.append(int(row[k-1]))
            training_data.append(row)


#print(len(raw_train_labels))

with open(filename2, 'r') as csvfile: 
    csvreader = csv.reader(csvfile) 
    fields = next(csvreader) 
    l = 0
    for row in csvreader: 
        if(l==0):
            l+=1
        else:
            k = len(row)
            raw_test_data.append(row[0:(k-1)])
            raw_test_labels.append(int(row[k-1]))
            testing_data.append(row)

#print(len(raw_test_labels))


with open(filename3, 'r') as csvfile: 
    csvreader = csv.reader(csvfile) 
    fields = next(csvreader) 
    l = 0
    for row in csvreader: 
        if(l==0):
            l+=1
        else:
            k = len(row)
            raw_val_data.append(row[0:(k-1)])
            raw_val_labels.append(int(row[k-1]))
            val_data.append(row)



def linearize(arr):
    m = len(arr)
    n = len(arr[0])
    o = 1
    #print(m)
    #print(n)
    #print(o)
    ans = np.zeros((m,n*o),dtype=int)
    i = 0
    while(i<m):
        a = arr[i]
        k = 0
        j1 = 0
        while(j1<n):
            j2 = 0
            while(j2 <o):
                ans[i][k] = arr[i][j1][j2]
                k += 1
                j2 += 1
            j1 += 1
        i+=1


    return(ans)


#new_train_data = linearize(raw_train_data)
#new_test_data = linearize(raw_test_data)

def modify(arr):
    m = len(arr)
    #n = arr.shape[1]
    #o = arr.shape[2]
    print(m)
    ans = np.array(arr).reshape(m,1)
    #print(n)
    #print(o)
    i = 0
    '''while(i<m):
        a = arr[i]
        ans[i] = a
        #print(a)
        i+=1'''
    return(ans)


new_train_labels = modify(raw_train_labels)
new_test_labels = modify(raw_test_labels)



from sklearn.ensemble import RandomForestClassifier

param1 = [50,150,250,350,450]

test_accs1 = []
train_accs1 = []
val_accs1 = []

for p1 in param1:

    clf = RandomForestClassifier(n_estimators = p1, max_features = 0.7, min_samples_split = 2)
    clf.fit(raw_train_data,raw_train_labels)

    #print("Accuracy on test set = "+str(clf.score(raw_test_data,raw_test_labels)))
    #print("Accuracy on train set = "+str(clf.score(raw_train_data,raw_train_labels)))
    #print("Accuracy on val set = "+str(clf.score(raw_val_data,raw_val_labels)))
    test_accs1.append(clf.score(raw_test_data,raw_test_labels))
    train_accs1.append(clf.score(raw_train_data,raw_train_labels))
    val_accs1.append(clf.score(raw_test_data,raw_test_labels))
print("p1 done...")
print(test_accs1)
print(train_accs1)
print(val_accs1)



param2 = [0.1,0.3,0.5,0.7,0.9]
test_accs2 = []
train_accs2 = []
val_accs2 = []

for p2 in param2:

    clf = RandomForestClassifier(n_estimators = 450, max_features = p2, min_samples_split = 2)
    clf.fit(raw_train_data,raw_train_labels)

    #print("Accuracy on test set = "+str(clf.score(raw_test_data,raw_test_labels)))
    #print("Accuracy on train set = "+str(clf.score(raw_train_data,raw_train_labels)))
    #print("Accuracy on val set = "+str(clf.score(raw_val_data,raw_val_labels)))
    test_accs2.append(clf.score(raw_test_data,raw_test_labels))
    train_accs2.append(clf.score(raw_train_data,raw_train_labels))
    val_accs2.append(clf.score(raw_test_data,raw_test_labels))
print("p2 done...")
print(test_accs2)
print(train_accs2)
print(val_accs2)


param3 = [2,4,6,8,10]
test_accs3 = []
train_accs3 = []
val_accs3 = []

for p3 in param3:

    clf = RandomForestClassifier(n_estimators = 450, max_features = 0.7, min_samples_split = p3)
    clf.fit(raw_train_data,raw_train_labels)

    #print("Accuracy on test set = "+str(clf.score(raw_test_data,raw_test_labels)))
    #print("Accuracy on train set = "+str(clf.score(raw_train_data,raw_train_labels)))
    #print("Accuracy on val set = "+str(clf.score(raw_val_data,raw_val_labels)))
    test_accs3.append(clf.score(raw_test_data,raw_test_labels))
    train_accs3.append(clf.score(raw_train_data,raw_train_labels))
    val_accs3.append(clf.score(raw_test_data,raw_test_labels))
print("p3 done...")
print(test_accs3)
print(train_accs3)
print(val_accs3)





def write_predictions(fname, arr):
    np.savetxt(fname, arr, fmt="%d", delimiter="\n")

#test_p = clf.predict(raw_test_data)
#write_predictions(outputfile,test_p)


'''
The optimal parameters obtained after thorough testing are :
N_estimators = 450
Max_features = 0.7
Min_samples_split = 2

'''