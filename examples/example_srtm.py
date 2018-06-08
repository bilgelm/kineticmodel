
# coding: utf-8

# In[85]:


get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')


# In[86]:


from kineticmodel import SRTM_Zhou2003, SRTM_Lammertsma1996, SRTM_Gunn1997


# In[87]:


import sys, os
sys.path.insert(0,os.pardir)
from tests.generate_test_data import generate_fakeTAC_SRTM


# In[88]:


import numpy as np
np.random.seed(0)

import scipy as sp
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[89]:


# generate noiseless fake data based on SRTM
BP = 0.5
R1 = 1.2
t, dt, TAC, refTAC = generate_fakeTAC_SRTM(BP, R1)

fig, ax = plt.subplots();
ax.plot(t, TAC, label='TAC');
ax.plot(t, refTAC, label='Reference TAC');
ax.set_xlabel('t');
ax.set_ylabel('Activity');
ax.set_title('Simulated data');
ax.legend();


# In[90]:


# Initialize SRTM Lammerstma 1996 model
mdl_lammertsma = SRTM_Lammertsma1996(t, dt, TAC, refTAC, time_unit='min')

# fit model
mdl_lammertsma.fit();

# get model results
mdl_lammertsma.results


# In[91]:


mdl_gunn = SRTM_Gunn1997(t, dt, TAC, refTAC, time_unit='min')

mdl_gunn.fit()

mdl_gunn.results


# In[307]:


# Initialize SRTM Zhou 2003 model
mdl_zhou = SRTM_Zhou2003(t, dt, TAC, refTAC, time_unit='min')

mdl_zhou.fit();

mdl_zhou.results


# In[296]:


# Generate noisy simulations by adding normal noise -- I don't think this is a good way
pct_noise = np.array([0, 5, 10, 15, 20, 25, 30])

TAC_matrix = TAC + np.random.normal(0,np.outer(TAC,pct_noise/100).T)


# In[297]:


fig, ax = plt.subplots();
ax.plot(t, TAC_matrix.T, label='');
ax.plot(t, TAC, 'k-', label='TAC');
ax.plot(t, refTAC, 'k--', label='Reference TAC');
ax.set_xlabel('t');
ax.set_ylabel('Activity');
ax.set_title('Simulated data');
ax.legend();


# Experiment using noisy TAC and noiseless reference TAC

# In[298]:


# Initialize SRTM Lammerstma 1996 model
mdl_lammertsma = SRTM_Lammertsma1996(t, dt, TAC_matrix, refTAC, time_unit='min')

# fit model
mdl_lammertsma.fit();

# get model results
mdl_lammertsma.results


# In[299]:


# Initialize SRTM Zhou 2003 model
mdl_zhou = SRTM_Zhou2003(t, dt, TAC_matrix, refTAC, time_unit='min')

mdl_zhou.fit();

mdl_zhou.results


# In[300]:


fig, axes = plt.subplots(1,2, figsize=(10,4));

axes[0].plot(pct_noise, mdl_lammertsma.results['BP'], '.', label='Lammertsma 1996');
axes[0].plot(pct_noise, mdl_zhou.results['BP'], '.', label='Zhou 2003 w/o spatial constraint');
axes[0].axhline(y=BP, color='k', linestyle='--');
axes[0].set_xlabel('% noise');
axes[0].set_ylabel('BP');
#axes[0].legend();

axes[1].plot(pct_noise, mdl_lammertsma.results['R1'], '.', label='Lammertsma 1996');
axes[1].plot(pct_noise, mdl_zhou.results['R1'], '.', label='Zhou 2003 w/o spatial constraint');
axes[1].axhline(y=R1, color='k', linestyle='--');
axes[1].set_xlabel('% noise');
axes[1].set_ylabel('R1');
axes[1].legend();

