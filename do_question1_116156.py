import numpy as np
import csv
import math

import sys

net_acc = 0
num_nodes = 0


def classify1(rows):
    dicti = {}
    l = len(rows)
    i = 0
    while(i<l):
        
        label = (rows[i])[-1]
        if(label not in dicti):
            dicti[label] = 0

        i += 1
        dicti[label] += 1


    return(dicti)

def deviation(rows):

    curr_dev = 1
    c_array = classify1(rows)
    l = len(rows)
    l1 = float(l)
    
    for lbl in c_array:

        p = float(c_array[lbl] / l1)

        p1 = p**2
        curr_dev = curr_dev - p1

    finel_dev = curr_dev
    return(finel_dev)



from time import time
def unique_vals(rows,c):
    l = []
    for r in rows:
        l.append(r[c])
    
    s = set(l)
    ####
    return(s)


def write_predictions(fname, arr):
    np.savetxt(fname, arr, fmt="%d", delimiter="\n")

class decision:
    

    def __init__(self,c,v):

        self.val = 0
        self.col = 0

        self.val = v
        self.col = c
        

    def row_val(self,x):
        
        val = x[self.col]
        if(isinstance(val,int) or isinstance(val,float)):
            if(self.val <= val):
                return(True)

            else:
                return(False)
            
        else:
            if(self.val == val):
                return(True)
            else:
                return(False)
            


def split(r,q):

    rt = []
    rf = []
    l = len(r)
    i = 0

    while(i<l):
        temp = r[i]
        if(q.row_val(temp) == True):
            rt.append(temp)
        elif(q.row_val(temp) == False):
            rf.append(temp)
        else:
            break
        i=i+1

    return rt,rf


def encaps(p):
    return(p*math.log(p))
def entropy(dev,l_node,r_node):
    
    l1 = len(l_node)
    l2 = len(r_node)
    l11 = float(l1)
    l22 = float(l2)
    p = l11/(l11+l22)
    p = encaps(p)

    ld = deviation(l_node)
    rd = deviation(r_node)

    k = dev - ((p*ld) + ((1-p)*rd))
    #k = k*math.log(k)
    return(k)

def get_vals(r,index):
    l = len(r)
    l1 = []
    i = 0
    for temp in r:
        #temp = r[i]
        l1.append(temp[index])

        i=i+1

    s = set(l1)

    return(s)


def find_best_split(rows):

    curr_dev = deviation(rows)

    d_best = None
    
    g_max = 0 
      
    l = len(rows[0])
    n_features = l-1

    i = 0

    for i in range(n_features): 

        values = get_vals(rows,i)
        

        for val in values:

            d = decision(i, val)
            rt, rf = split(rows,d)

            if(len(rf) == 0):
                continue

            if(len(rt) == 0):
                continue

            g_new = entropy(curr_dev,rt,rf)


            if(g_max <= g_new):
                g_max = g_new
                d_best = d
        i = i + 1

    return g_max, d_best


def find_best_split1(rows):

    curr_dev = deviation(rows)

    d_best = None
    
    g_max = 0 
      
    l = len(rows[0])
    n_features = l-1

    i = 0

    for i in range(n_features): 

        values = get_vals(rows,i)
        

        for val in values:

            d = decision(i, val)
            rt, rf = split(rows,d)

            if(len(rf) == 0):
                continue

            if(len(rt) == 0):
                continue

            g_new = entropy(curr_dev,rt,rf)


            if(g_max <= g_new):
                g_max = g_new
                d_best = d
        i = i + 1

    return g_max, d_best



############## leaf node
class LeafNode:
    ############
    def __init__(self,r):
        ############
        self.preds = classify1(r)


def raw_preds(predicted_all):

    l = predicted_all.values()
    t = sum(l)
    t1 = float(t)
    
    p = {}
    k = predicted_all.keys()
    global net_acc

    for key in k:
        net_acc += predicted_all[key]/t1

        p[key] = int((predicted_all[key]/t1) * 100)
        

    return(p)



####### decision node
class Decision_Node:

    def __init__(self,tb,fb,d):

        ########## for max branching
        self.tb = tb

        self.fb = fb

        self.decision = d
        global num_nodes
        num_nodes += 1
        '''if(num_nodes > 200):
            self.false_branch = true_branch ## this commented 30% for 1000 with this 5%'''

def classify(x,node):
    
    if isinstance(node, LeafNode):
        return(node.preds)

    else:
        d = node.decision.row_val(x)

        if(d == True):
            return classify(x,node.tb)
        else:
            return classify(x,node.fb)



def build_tree(data):
    
    g,d = find_best_split(data)

    global num_nodes
    if(g == 0):
        return LeafNode(data)

    rt,rf = split(data,d)

    
    if(num_nodes > -1):
        tb = build_tree(rt)


    if(num_nodes >-1):
        fb = build_tree(rf)


    if(num_nodes > -1):
        return Decision_Node(tb,fb,d)


def build_treep(data):
    
    g,d = find_best_split1(data)

    global num_nodes
    if(g == 0):
        return LeafNode(data)

    rt,rf = split(data,d)

    
    if(num_nodes > -1):
        tb = build_treep(rt)


    if(num_nodes >-1):
        fb = build_treep(rf)


    if(num_nodes > -1):
        return Decision_Node(tb,fb,d)










st = time()

with_pruning = int(sys.argv[1])

filename1 = sys.argv[2]
filename2 = sys.argv[4]
filename3 = sys.argv[3]

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

#print(len(raw_val_labels))

s = 100
#print(s)
if(with_pruning == 1):
    my_tree = build_tree(training_data[0:s])

else:
    my_tree = build_treep(training_data[0:s])

predictions = []  ## test
predictions1 = []  ## train
predictions2 = []  ## val


def max_prob(d):
    m = 0
    k = 1
    for g in d.keys():
        if(d[g] > m):
            m = d[g]
            k = int(g)
    return(k)


predictions_for_test_data = {}

def check_accuracy(a,b):
    l = len(a)
    t = 0
    i = 0
    while(i<l):
        if(a[i]==b[i]):
            t+=1
        i+=1
    #print("num matches : "+str(t)+" out of "+str(l))

    return(float(t)/float(l))

def findt(ts,tr,l):
    best_tree = build_tree(training_data[0:10])
    best_acc = 0

    st = [100,500,1000,2000,3000]
    for s in st:
        #print("for s = "+str(s))
        mt = build_tree(training_data[0:s])
        temp_preds = []

        for t in ts:
            c = classify(t,mt)
            l1 = raw_preds(c)
            temp = max_prob(l1)
            temp_preds.append(temp)

        #print(len(temp_preds))
        #print(len(l))
        a = check_accuracy(temp_preds,l)
        if(a>best_acc):
            best_acc = a
            best_tree = mt

    return(best_tree)
        



my_tree = findt(val_data,training_data,raw_val_labels)

for row in testing_data:
    c = classify(row,my_tree)
    #print(c)
    
    l = raw_preds(c)
    #print(l)
    temp = max_prob(l)
    #print(temp)
    predictions.append(temp)

for row in training_data:
    c = classify(row,my_tree)
    temp = max_prob(raw_preds(c))
    predictions1.append(temp)


for row in val_data:
    c = classify(row,my_tree)
    temp = max_prob(raw_preds(c))
    predictions2.append(temp)




#print(net_acc/len(testing_data))


filename4 = sys.argv[5]
write_predictions(filename4,predictions)




#print("Number of nodes = " + str(num_nodes))
#print("Accuracy on test set = "+ str(check_accuracy(predictions,raw_test_labels)))
#print("Accuracy on training set = "+ str(check_accuracy(predictions1,raw_train_labels)))
#print("Accuracy on validation set = "+ str(check_accuracy(predictions2,raw_val_labels)))

et = time()
#print("Total time taken = "+str(et-st))


#print(predictions[:10])
#print(raw_test_labels[:10])