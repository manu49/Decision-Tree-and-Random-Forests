## 2 a
## comment out f4, test labels aren't to be read
## remove evaluate function

import numpy as np

from scipy.stats import truncnorm
import sys


from time import time


#######################
dicti = {}
j = 0
while(j<10):
    dicti[j] = 0
    j += 1
#######################


def normalize(mean,std_dev,low,up):
    l= (low/std_dev) - (mean/std_dev)
    h = (up/std_dev) - (mean/std_dev)
    # print(l)
    # print(h)
    return truncnorm(l,h,loc=mean,scale=std_dev)


def sigmoid(x):
    z = 1/(1 + np.exp((-1)*x))
    return(z)




def write_predictions(fname, arr):
    np.savetxt(fname, arr, fmt="%d", delimiter="\n")

class CNN:
        
    
    def __init__(self,network_structure,alpha):  

        self.structure = network_structure
        self.learning_rate = alpha
        
        self.bias = None
        self.total_layers = len(network_structure)

        self.weights = []

        total_layers = len(network_structure)
        i = 1
        while(i<total_layers):
            fan_in = network_structure[i-1]
            fan_out = network_structure[i]

            temp1 = np.sqrt(fan_in)

            combinations = fan_in * fan_out
            radian = 1/temp1

            X_new = normalize(2,1,(-1)*radian,radian)

            combinations = fan_out * fan_in
            weight_layer_i_temp = X_new.rvs(combinations)

            weight_layer_i = weight_layer_i_temp.reshape((fan_out,fan_in))

            self.weights.append(weight_layer_i)


            i = i +1

    def evaluate1(self,X):
        preds = []
        for x in X:
            d = self.run(x)
            cl = d.argmax()
            preds.append(cl)

        return(preds)


    def evaluate(self,X,y):

        global dicti

        m = 0
        i = 0

        for x in X:
            d = self.run(x)
            cl = d.argmax()
            #print(cl)
            dicti[cl] +=1
            if(cl == y[i]):
                m = m + 1
            i= i + 1


        acc = float(m)/X.shape[0]
        return(acc)

        
    def train(self,X,y):

        ## FORWARD PROPAGATION
                                  

        X_reshaped = np.array(X,ndmin=2).T
        Y_reshaped = np.array(y,ndmin=2).T

        all_vecs = [X_reshaped]
        
        
        i = 0
        while(i <= self.total_layers - 2):
            temp_x = [X_reshaped]


            
            features = all_vecs[-1]
            w = self.weights[i]
            

            n_v = sigmoid(np.dot(w,features))
            
            all_vecs.append(n_v)    
            i = i + 1

        ## BACK PROPAGATION
        err = (Y_reshaped - n_v)
        ## initial error
        
        j = self.total_layers - 1
        alpha = self.learning_rate
        while(j >= 1):

            a = all_vecs[j-1].T
            n_v = all_vecs[j]
            

            temp1 = err*n_v*(1.0-n_v) 

            delta = np.dot(temp1,a)
                
            self.weights[j-1] = self.weights[j-1] + alpha*delta

            w = self.weights[j-1].T
            # w = a.T
            # w = self.weights[j-1]
            err = np.dot(w,err)
            j = j - 1
            
    def train_batch(self,B,batch_size):
        for b in B:
            self.train(b[0],b[1])

               
    
    def run(self,X):
        
        a_initial = np.array(X, ndmin=2).T
        a = a_initial

        i = 1
        
        while(i <= self.total_layers-1):

            w = self.weights[i-1]

            a1 = sigmoid(np.dot(w,a))
            a = a1
            i = i + 1
  
    
        return(a)
########################################## main execution

start_time = time()

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

batch_size = int(sys.argv[5])
hidden_layers_list = sys.argv[6]
activation_type = sys.argv[7]


hidden_layer_list_parsed = map(int,hidden_layers_list.strip('[]').split(','))

m = 28*28
structure_list = [m]
#num_hidden_layers = len(hidden_layer_list_parsed)

i = 0
for h in hidden_layer_list_parsed:
    #h = hidden_layer_list_parsed[i]
    #print(str(h))
    structure_list.append(h)
    i+=1

structure_list.append(10)

print(structure_list)


model = CNN(structure_list,alpha=0.001)

print("model initialised")

## 50 : 82.89 expit
## 100 : 86.18 expit
## 500 : 87.8 expit
    
    
'''for i in range(new_train_data.shape[0]):
    model.train(new_train_data[i],one_hot_train_labels[i])'''

for i in range(new_train_data.shape[0]):
    model.train(new_train_data[i],one_hot_train_labels[i])



'''num_of_batches = int(new_train_data.shape[0]/batch_size)
i = 0
while(i<num_of_batches):
    st = i*batch_size
    B = []
    j = 0
    while(j<batch_size):
        x = new_train_data[st+j]
        y = new_train_labels[st+j]
        l = [x,y]
        B.append(l)
        j += 1

    ## one batch made
    model.train_batch(B,batch_size)
    i = i + 1'''




'''
corrects, wrongs = ANN.evaluate(new_train_data,new_train_labels)
print("accuracy train: ", corrects / ( corrects + wrongs))
'''

f5 = sys.argv[4]
f = open(f5,'w')
f.close()


preds = model.evaluate1(new_test_data)
write_predictions(f5,preds)

end_time = time()

acc = model.evaluate(new_test_data,new_test_labels)
acc1 = model.evaluate(new_train_data,new_train_labels)
print("accuracy on test set = " + str(acc))
print("accuracy on train set = "+ str(acc1))
print("total time taken = "+str(end_time-start_time)+" seconds")


j = 0
while(j<10):
    print(dicti[j])
    j+=1



####### notes ########


######################