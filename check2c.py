import matplotlib.pyplot as plt 

arr = [1,10,50,100,500]
times = [51.77,54.07,89.29,111.42,244.56]
test_accs = [13,24,88,92,95]
train_accs = [22,36,93,95,97]

plt.plot(arr,times,label = 'time taken in seconds')
plt.xlabel('number of hidden units')
#plt.ylabel('time taken') 
#plt.title('time taken for training models with different hidden units')
#plt.savefig('2b_time.png')
#plt.legend()

plt.plot(arr,test_accs,label = 'test set accuracy for training models with different hidden units')
'''plt.xlabel('number of hidden units') 
plt.ylabel('accuracy on test set') 

plt.savefig('2b_acc2.png')'''



plt.plot(arr,train_accs,label='training set accuracy for training models with different hidden units' )
#plt.xlabel('number of hidden units') 
#plt.ylabel('accuracy on training set') 
#plt.legend()
plt.savefig('2c.png')