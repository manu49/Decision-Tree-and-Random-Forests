from sklearn.neural_network import MLPClassifier
import sys
import numpy as np
import math

from time import time

def read_train_data(filename):
    train_data = np.load(filename)
    #print(type(train_data))
    return(train_data)

def read_train_labels(filename):
    train_labels = np.load(filename)
    #print(type(train_data))
    return(train_labels)

#f1 = "kannada_digits/neural_network_kannada/X_train.npy"
f1 = sys.argv[1]
train_data = read_train_data(f1)
#print(train_data[0].shape)
#print(train_data[1].shape)
#f2 = "kannada_digits/neural_network_kannada/y_train.npy"
f2 = sys.argv[2]
train_labels = read_train_labels(f2)

#f3 = "kannada_digits/neural_network_kannada/X_test.npy"
f3 = sys.argv[3]
test_data = read_train_data(f3)
#print(train_data[0].shape)
#print(train_data[1].shape)
f4 = "kannada_digits/neural_network_kannada/y_test.npy"
test_labels = read_train_labels(f4)

print("raw data done...")


def linearize(arr):
    m = arr.shape[0]
    n = arr.shape[1]
    o = arr.shape[2]
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


new_train_data = linearize(train_data)
new_test_data = linearize(test_data)

#print(new_train_data[0])
#print(new_train_data.shape)
#print(type(new_train_data))


def modify(arr):
    m = arr.shape[0]
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


new_train_labels = modify(train_labels)
new_test_labels = modify(test_labels)
##########################################




def make_one_hot(arr):
    l = len(arr)
    f = np.zeros((l,10),dtype=int)
    temp_arr = np.zeros((10),dtype=int)
    i = 0
    for x in arr:
        temp_arr[x] = 1
        f[i] = temp_arr
        temp_arr[x] = 0
        i+=1

    return(f)



one_hot_train_labels = make_one_hot(train_labels)
#print(one_hot_train_labels[0:15])
print("one hot encoding done")

st = time()
clf = MLPClassifier(random_state=1, max_iter=300).fit(new_train_data,new_train_labels)



print(clf.score(new_test_data,new_test_labels))
print(clf.score(new_train_data,new_train_labels))
et = time()
print("time taken = "+str(et-st))