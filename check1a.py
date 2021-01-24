import matplotlib.pyplot as plt 

'''x = [56,210,412,]
tr = [47.17,44.09,31.87,]
ts = [47.03,43.98,31.56,]
vl = [47.28,44.30,31.80,]
'''

x = [412,1708,3558,27654,80000]
ts = [32.69,43.07,48.56,63.88,77.93]
tr = [35.32,54.76,65.65,83.12,93.56]
vl = [33.09,45.44,57.94,66.38,84.53]

plt.plot(x,ts,label='test acc')
plt.plot(x,tr,label='train acc')
plt.plot(x,vl,label='val acc')
plt.xlabel('number of nodes in DT')
plt.ylabel('accuracies on different sets')

plt.legend()

plt.savefig('1a.png')