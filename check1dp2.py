import matplotlib.pyplot as plt 

x = [0.1,0.3,0.5,0.7,0.9]
a1 = [0.9433606529150453, 0.9602345122077206, 0.9634457109404594, 0.9639708667653845, 0.9633165742621991]
a2 = [1.0, 1.0, 1.0, 1.0, 1.0]
a3 = [0.9513, 0.9524, 0.9554, 0.9575, 0.961]


plt.plot(x,a1,label='test acc')
plt.plot(x,a2,label='train acc')
plt.plot(x,a3,label='val acc')

plt.legend()
plt.xlabel('max features')
plt.ylabel('accuracies')

plt.savefig('1dp2.png')