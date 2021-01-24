import matplotlib.pyplot as plt 

x = [50,150,250,350,450]

a1 = [0.9625245359688694, 0.9631443920245187, 0.9633596198216192, 0.964211921898137, 0.9641430490030648]
a2 = [0.9999785097427616, 1.0, 1.0, 1.0, 1.0]
a3 = [0.988, 0.989, 0.990, 0.992, 0.992]

plt.plot(x,a1,label='test acc')
plt.plot(x,a2,label='train acc')
plt.plot(x,a3,label='val acc')

plt.legend()
plt.xlabel('number of estimators')
plt.ylabel('accuracies')

plt.savefig('1dp1.png')