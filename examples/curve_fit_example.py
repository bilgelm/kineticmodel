
# coding: utf-8

# In[28]:

# from https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# In[29]:

def func(x, a, b, c):
     return a * np.exp(-b * x) + c


# In[30]:

xdata = np.linspace(0, 4, 50)
y = func(xdata, 2.5, 1.3, 0.5)
np.random.seed(1729)
y_noise = 0.2 * np.random.normal(size=xdata.size)
ydata = y + y_noise
plt.plot(xdata, ydata, 'b-', label='data')


# In[31]:

popt, pcov = curve_fit(func, xdata, ydata)
popt


# In[32]:

plt.plot(xdata, func(xdata, *popt), 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))


# In[33]:

popt, pcov = curve_fit(func, xdata, ydata, bounds=(0, [3., 1., 0.5]))
popt


# In[34]:

plt.plot(xdata, func(xdata, *popt), 'g--', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))


# In[35]:

popt, pcov = curve_fit(func, xdata, ydata, bounds=(0, [3., 1., 0.5]),p0=([1.0 , 1.0, 0.4]))
popt


# In[36]:

plt.plot(xdata, func(xdata, *popt), 'y--', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))


# In[37]:

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


# In[ ]:




# In[ ]:



