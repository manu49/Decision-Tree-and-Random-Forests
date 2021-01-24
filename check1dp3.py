import matplotlib.pyplot as plt 

x = [2,4,6,8,10]
a1 = [0.9637298116326319, 0.9629808188987224, 0.9618702434656841, 0.9608371500396019, 0.9591842005578705]
a2 = [1.0, 0.9999293891547881, 0.998925487138081, 0.9972062665590107, 0.9950480307249278]
a3 = [0.9837, 0.9830, 0.9819, 0.9808, 0.9792]

plt.plot(x,a1,label='test acc')
plt.plot(x,a2,label='train acc')
plt.plot(x,a3,label='val acc')

plt.legend()
plt.xlabel('minimum samples split')
plt.ylabel('accuracies')

plt.savefig('1dp3.png')