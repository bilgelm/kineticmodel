
# coding: utf-8

# In[1]:


# from https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# In[2]:


def func(x, a, b, c):
     return a * np.exp(-b * x) + c


# In[3]:


xdata = np.linspace(0, 4, 50)
y = func(xdata, 2.5, 1.3, 0.5)
np.random.seed(1729)
y_noise = 0.2 * np.random.normal(size=xdata.size)
ydata = y + y_noise
plt.plot(xdata, ydata, 'b-', label='data')


# In[5]:


popt, pcov = curve_fit(func, xdata, ydata)
popt


# In[6]:


plt.plot(xdata, func(xdata, *popt), 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))


# In[37]:


popt, pcov = curve_fit(func, xdata, ydata, p0=([10 , 10, 9]))
popt


# In[38]:


plt.plot(xdata, func(xdata, *popt), 'g--', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))


# In[40]:


popt, pcov = curve_fit(func, xdata, ydata, p0=([2.0 , 1.3, 0.9]))
popt


# In[41]:


plt.plot(xdata, func(xdata, *popt), 'r--', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))


# In[42]:


plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

