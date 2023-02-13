#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from scipy import signal
import sklearn 
import seaborn as sns
from scipy.stats import norm


# In[4]:


pd.read_csv('https://raw.githubusercontent.com/BenedettaPasqualettounivr/Probability-project/main/India_Menu.csv?token=GHSAT0AAAAAAB6IB3KAZ6POJ4X4SZVHGCXGY7KWJ5A')


# In[5]:


customer_state = ['Ordering the sandwich','Waiting for your sandwich','Leaving The counter']
states = {'O':customer_state[0],'M':customer_state[1],'L':customer_state[2]}


# In[6]:


#One customer one sandwich


# In[7]:


mu, sigma = 5,1
def one_sandwich_one_cust():
    start = states['O']
    print(start+'\n')
    ordering_time = 0.5
    first_state = states['M']
    print(first_state+'\n')
    waiting_time = 0
    k = 0
    while k == 0:
        p = norm.cdf(waiting_time, loc=mu, scale=sigma)
        k = np.random.choice([0,1],p = [1-p,p])
        waiting_time = waiting_time+0.5
        if k == 0:
            print('Sandwich is brewing... \n')
    print('Your sandwich is ready! \n')
    print(states['L']+'\n')
    print('Waiting time is = %.2f'%(waiting_time))
    return waiting_time


# In[8]:


one_sandwich_one_cust()


# In[9]:


#One customer multiple sandwiches


# In[10]:


#It isn't know what is the most chosen sandwich, let’s just make a hypotetical distribution:


# In[13]:


kind_of_sandwich = np.array(pd.read_csv('https://raw.githubusercontent.com/BenedettaPasqualettounivr/Probability-project/main/India_Menu.csv?token=GHSAT0AAAAAAB6IB3KAZ6POJ4X4SZVHGCXGY7KWJ5A')['Menu Items'])
p_start = []
for i in range(len(kind_of_sandwich)):
    p_start.append(np.random.choice(np.arange(50,100)))
p_start = np.array(np.array(list(np.array(p_start)/sum(p_start))))


# In[14]:


#So, the probability distribution is


# In[15]:


sandwich_picked = []
for i in range(10000):
    sandwich_picked.append(np.random.choice(range(0,len(kind_of_sandwich)),p=p_start))
sns.displot(sandwich_picked)


# In[16]:


sandwich_data = pd.DataFrame(kind_of_sandwich,columns=['State 1'])
mu_list = []
var_list = []
for i in range(len(sandwich_data)):
    mu_list.append(np.random.choice(np.linspace(3,6,1000)))
    var_list.append(np.random.choice(np.linspace(0.1,1.5,1000)))
sandwich_data[r'$\mu$']=mu_list
sandwich_data[r'$\sigma$']=var_list
sandwich_data[r'$p$'] = p_start
sandwich_data.head()


# In[17]:


def random_sandwich_one_cust():
    start = states['O']
    print(start+'\n')
    ordering_time = 0.5
    first_state = states['M']
    chosen_i = np.random.choice(range(0,len(kind_of_sandwich)),p=p_start)
    print('Ordering sandwich %s'%(kind_of_sandwich[chosen_i]))
    print(first_state+'\n')


    mu_i, var_i = sandwich_data[r'$\mu$'].loc[chosen_i], sandwich_data[r'$\sigma$'].loc[chosen_i]
    waiting_time = 0
    k = 0
    while k == 0:
        p = norm.cdf(waiting_time, loc=mu_i, scale=var_i)
        k = np.random.choice([0,1],p = [1-p,p])
        waiting_time = waiting_time+0.5
        if k == 0:
            print('Sandwich is brewing... \n')
    print('Your sandwich is ready! \n')
    print(states['L']+'\n')
    print('Waiting time is = %.2f'%(waiting_time))
    return waiting_time
random_sandwich_one_cust()


# In[18]:


#“How much time does it usually take to grab a whatever sandwich at Mc Donald's?”


# In[19]:


waiting_time_list = []
for i in range(100):
    waiting_time_list.append(random_sandwich_one_cust())
plt.figure(figsize=(20,10))
plt.subplot(2,1,1)
sns.histplot(waiting_time_list,fill=True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Waiting time (minutes)',fontsize=14)
plt.ylabel('Count',fontsize=14)
plt.subplot(2,1,2)
sns.kdeplot(waiting_time_list,fill=True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Waiting time (minutes)',fontsize=14)
plt.ylabel('Probability Density Function (PDF)',fontsize=14)


# In[20]:


#Multiple Customers Multiple sandwich


# In[21]:


def random_sandwich_multiple_cust(cust=2,num_baristas =5):
    time_of_process = []
    baristas = np.zeros(num_baristas)
    q = 0
    ordering_time = 0
    for c in range(cust):
        start = states['O']
        print('Customer number %i is ordering'%(c))
        ordering_time+=0.5
        if sum(baristas)!=num_baristas:
            print('There is at least one free baristas! :)')
            waiting_time = random_sandwich_one_cust()
            time_of_process.append(waiting_time+ordering_time)
            baristas[q] = 1
            q = q + 1 
        if len(time_of_process)==cust:
            break
        if sum(baristas)==num_baristas:
            print('All the baristas are busy :(')
            print('You have to wait an additional %i minutes until they can start making your sandwich' %(min(time_of_process)))
            waiting_time = min(time_of_process)+random_sandwich_one_cust()+ordering_time
            baristas[num_baristas-1]=0
            time_of_process.append(waiting_time)
            q = q-1
        if len(time_of_process)==cust:
            break
    print('The waiting time for each customer is:')
    print(time_of_process)
    return time_of_process


# In[22]:


random_sandwich_multiple_cust(2,5)


# In[23]:


random_sandwich_multiple_cust(10,2)


# In[ ]:




