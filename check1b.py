import matplotlib.pyplot as plt 

'''x = [56,210,412,]
tr = [47.17,44.09,31.87,]
ts = [47.03,43.98,31.56,]
vl = [47.28,44.30,31.80,]
'''

x = [412,1708,3558,27654,80000]
ts = [35.76,47.45,53.22,69.65,84.23]
tr = [38.65,59.75,71.61,91.62,98.60]
vl = [36.69,52.74,63.21,72.97,87.43]

plt.plot(ts,x,label='test acc')
plt.plot(tr,x,label='train acc')
plt.plot(vl,x,label='val acc')
plt.xlabel('number of nodes in DT')
plt.ylabel('accuracies on different sets')

plt.legend()

plt.savefig('1b.png')