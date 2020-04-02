#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import pandas as pd 
import random
import math
import datetime 
from dateutil.parser import parse
from sklearn.impute import SimpleImputer


# # CORONA VIRUS ANALYSIS

# In[4]:


data = pd.read_csv("C:\\Users\\Admin-pc\\Downloads\\covid_19_data.csv")
data.tail()


# In[5]:


data.info()

adjusted_unique_dates = list(data['Last Update'].unique())
adjusted_unique_dates.sort()
adjusted_unique_dates

data.drop_duplicates(inplace=True)

data.replace('China', 'Mainland China', inplace=True)

data[data.Country == 'Mainland China']

excess_dates = []
for i in range(len(adjusted_unique_dates)):
    # assume the number of coronavirus cases, deaths, and recover increases over time 
    if i != 0:
        current_day_cases = data[data['Last Update']==adjusted_unique_dates[i]].Confirmed.sum()
        prev_day_cases = data[data['Last Update']==adjusted_unique_dates[i-1]].Confirmed.sum()
        current_day_deaths = data[data['Last Update']==adjusted_unique_dates[i]].Deaths.sum()
        prev_day_deaths = data[data['Last Update']==adjusted_unique_dates[i-1]].Deaths.sum()
        current_day_recovered = data[data['Last Update']==adjusted_unique_dates[i]].Recovered.sum()
        prev_day_recovered = data[data['Last Update']==adjusted_unique_dates[i-1]].Recovered.sum()
        
        if(current_day_cases < prev_day_cases or current_day_deaths < prev_day_deaths or current_day_recovered < prev_day_recovered):
            excess_dates.append(adjusted_unique_dates[i])
            # swap the current date with the previous date, it will get removed later
            temp = adjusted_unique_dates[i]
            adjusted_unique_dates[i] = adjusted_unique_dates[i-1]
            adjusted_unique_dates[i-1] = temp
            
for i in excess_dates:
    adjusted_unique_dates.remove(i)

world_cases = []
deaths = [] 
mortality_rate = []
recovered = [] 

for i in adjusted_unique_dates:
    confirmed_sum = data[data['Last Update']==i].Confirmed.sum()
    death_sum = data[data['Last Update']==i].Deaths.sum()
    recovered_sum = data[data['Last Update']==i].Recovered.sum()
    world_cases.append(confirmed_sum)
    deaths.append(death_sum)
    mortality_rate.append(death_sum/confirmed_sum)
    recovered.append(recovered_sum)


# In[6]:


plt.figure(figsize=(20, 12))
plt.plot(adjusted_unique_dates, world_cases)
plt.title('# of Coronavirus Cases Over Time', size=30)
plt.xlabel('Time', size=30)
plt.ylabel('# of Cases', size=30)
plt.xticks(rotation=50, size=15)
plt.show()


# In[7]:


plt.figure(figsize=(20, 12))
plt.plot(adjusted_unique_dates, deaths, color='red')
plt.title('# of Coronavirus Deaths Over Time', size=30)
plt.xlabel('Time', size=30)
plt.ylabel('# of Deaths', size=30)
plt.xticks(rotation=50, size=15)
plt.show()


# In[8]:


mean_mortality_rate = np.mean(mortality_rate)
plt.figure(figsize=(20, 12))
plt.plot(adjusted_unique_dates, mortality_rate, color='orange')
plt.axhline(y = mean_mortality_rate,linestyle='--', color='black')
plt.title('Mortality Rate of Coronavirus Over Time', size=30)
plt.legend(['mortality rate', 'y='+str(mean_mortality_rate)])
plt.xlabel('Time', size=30)
plt.ylabel('Mortality Rate', size=30)
plt.xticks(rotation=50, size=15)
plt.show()


# In[9]:


plt.figure(figsize=(20, 12))
plt.plot(adjusted_unique_dates, recovered, color='green')
plt.title('# of Coronavirus Cases Recovered Over Time', size=30)
plt.xlabel('Time', size=30)
plt.ylabel('# of Cases', size=30)
plt.xticks(rotation=50, size=15)
plt.show()


# In[10]:


plt.figure(figsize=(20, 12))
plt.plot(adjusted_unique_dates, deaths, color='red')
plt.plot(adjusted_unique_dates, recovered, color='green')
plt.legend(['death', 'recovered'], loc='best', fontsize=20)
plt.title('# of Coronavirus Cases', size=30)
plt.xlabel('Time', size=30)
plt.ylabel('# of Cases', size=30)
plt.xticks(rotation=50, size=15)
plt.show()


# In[11]:


plt.figure(figsize=(20, 12))
plt.plot(recovered, deaths)
plt.title('# of Coronavirus Deaths vs. # of Coronavirus Recoveries', size=30)
plt.xlabel('# of Coronavirus Recoveries', size=30)
plt.ylabel('# of Coronavirus Deaths', size=30)
plt.xticks(size=15)
plt.show()


# In[13]:


unique_provinces =  data['Province/State'][data.Confirmed > 0].unique()
unique_provinces

province_confirmed_cases = []
for i in unique_provinces:
    province_confirmed_cases.append(data[data.Confirmed>0][data['Province/State']==i].Confirmed.max())

nan_indices = [] 

# handle nan if there is any, it is usually a float: float('nan')
for i in range(len(unique_provinces)):
    if type(unique_provinces[i]) == float:
        nan_indices.append(i)

unique_provinces = list(unique_provinces)
province_confirmed_cases = list(province_confirmed_cases)

for i in nan_indices:
    unique_provinces.pop(i)
    province_confirmed_cases.pop(i)

# number of cases per country/region
unique_countries = data[data.Confirmed>0]['Country'].unique()
unique_countries.sort()
unique_countries

# find unique dates
unique_dates = list(data['Last Update'].unique())
unique_dates.sort()
unique_dates

country_confirmed_cases = []
latest_date = adjusted_unique_dates[-1]
for i in unique_countries:   
    if i == 'Mainland China':
        country_confirmed_cases.append(data[data['Country']==i][data['Last Update']==latest_date].Confirmed.sum()) 
    else:
        index = -1
        while(True):
            if(len(data[data['Country']==i][data['Last Update']==unique_dates[index]])>0):
                country_confirmed_cases.append(data[data['Country']==i][data['Last Update']==unique_dates[index]].Confirmed.sum()) 
                break
            else:
                index -= 1

# number of cases per country/region
for i in range(len(unique_countries)):
    print(f'{unique_countries[i]}: {country_confirmed_cases[i]} cases')


# In[14]:


# number of cases per province/state/city

for i in range(len(unique_provinces)):
    print(f'{unique_provinces[i]}: {province_confirmed_cases[i]} cases')


# In[17]:


plt.figure(figsize=(45, 45))
plt.barh(unique_countries, country_confirmed_cases)
plt.title('# of Coronavirus Confirmed Cases in Countries/Regions')
plt.show()


# In[18]:


plt.figure(figsize=(50, 50))
plt.barh(unique_provinces, province_confirmed_cases)
plt.title('# of Coronavirus Confirmed Cases in Provinces/States')
plt.show()


# # PIE CHART CONFIRMED CASES COUBTRYWISE AND PROVINCE WISE

# In[21]:


c = random.choices(list(mcolors.CSS4_COLORS.values()),k = len(unique_countries))
plt.figure(figsize=(40,20))
plt.pie(country_confirmed_cases, colors=c)
plt.legend(unique_countries, loc='best')
plt.show()


# In[22]:


c = random.choices(list(mcolors.CSS4_COLORS.values()),k = len(unique_countries))
plt.figure(figsize=(50,20))
plt.pie(province_confirmed_cases, colors=c)
plt.legend(unique_provinces, loc='best')
plt.show()


# In[5]:


#Correlation between feature over the time
covid= pd.read_csv("C:\\Users\\Admin-pc\\Downloads\\covid_19_data.csv")


# In[6]:


covid.head()


# In[7]:


mis = covid.isnull().sum()
mis[mis>0]


# In[14]:


imputer = SimpleImputer(strategy='constant')#here I use constant because I cannot put another Province/State
#that we do not know or that does not correspond to his country/region  
impute_covid = pd.DataFrame(imputer.fit_transform(covid), columns=covid.columns)
impute_covid.head()


# In[15]:


impute_covid['Date'] = pd.to_datetime(impute_covid['Date'])
impute_covid['Last Update'] = pd.to_datetime(impute_covid['Last Update'])
impute_covid['Confirmed'] = pd.to_numeric(impute_covid['Confirmed'], errors='coerce')
impute_covid['Recovered'] = pd.to_numeric(impute_covid['Recovered'], errors='coerce')
impute_covid['Deaths'] = pd.to_numeric(impute_covid['Deaths'], errors='coerce')


# In[16]:


impute_covid.head(3)


# In[17]:


impute_covid['active_confirmed'] = impute_covid['Confirmed'].values - (impute_covid['Deaths'].values+impute_covid['Recovered'].values)


# In[18]:


impute_covid.isnull().sum()[impute_covid.isnull().sum()>0]


# # CORRELATION BETWEEN FEATURES

# In[19]:


#Correlation between feature over the time
impute_covid.corr()


# In[22]:


features = [['Confirmed', 'Deaths'], ['Confirmed', 'Recovered'], ['Recovered', 'Deaths'],             ['Confirmed', 'active_confirmed']]
values = [[impute_covid['Confirmed'], impute_covid['Deaths']],          [impute_covid['Confirmed'], impute_covid['Recovered']],          [impute_covid['Recovered'], impute_covid['Deaths']],          [impute_covid['Confirmed'], impute_covid['active_confirmed']]]


# # FEATURES CURVE

# In[24]:



import seaborn as sns 
import statsmodels as sm

fig = plt.figure(figsize=(20.5,10.5))
fig.subplots_adjust(hspace=0.2, wspace=0.1)
for i in range(1,5):
    ax = fig.add_subplot(2, 2, i)
    col = features[i-1]
    val = values[i-1]
    ax.scatter(val[0], val[1])
    ax.set_xlabel(col[0])
    ax.set_ylabel(col[1])
    ax.set_title('Feature curves')
plt.show()


# In[25]:


start_date = impute_covid.Date.min()
end_date = impute_covid.Date.max()
print('Novel Covid-19 information:\n 1. Start date = {}\n 2. End date = {}'.format(start_date, end_date))


# In[26]:


worldwide = impute_covid[impute_covid['Date'] == end_date]


# In[27]:


nb_country = len(worldwide['Country'].value_counts()) # number country
worldwide['Country'].value_counts()


# In[28]:


world = worldwide.groupby('Country').sum()
world = world.sort_values(by=['Confirmed'], ascending=False)
world.head()


# # REPORT TILL 30 MARCH 

# In[29]:


print('================ Worldwide report ===============================')
print('== Information to {} on novel COVID-19 =========\n'.format(end_date))
print('Tota confirmed: {}\nTotal Deaths: {}\nTotal Recovered: {}\nTotal active confirmed: {}\nTotal country Recorded: {} \n'.format(worldwide.Confirmed.sum(), worldwide.Deaths.sum(), worldwide.Recovered.sum(), worldwide.active_confirmed.sum(),                                     nb_country))
print('==================================================================')


# # TOTAL CASES

# In[32]:


world.Confirmed.plot(kind='bar', title= 'novel Covid-19 in the Worldwide', figsize=(40,20), logy=True,legend=True)
plt.ylabel('Total Cases')


# # TOTAL RECOVERED

# In[33]:


world.Recovered.plot(kind='bar', title= 'novel Covid-19 in the Worldwide', figsize=(40,20), logy=True,                     colormap='Greens_r', legend=True)
plt.ylabel('Total Recovered')


# # TOTAL DEATHS

# In[34]:


world.Deaths.plot(kind='bar', title= 'novel Covid-19 in the Worldwide', figsize=(40,20), logy=True,                     colormap='Reds_r', legend=True)
plt.ylabel('Total Deaths')


# # TOTAL ACTIVE CASES

# In[35]:


world.active_confirmed.plot(kind='bar', title= 'novel Covid-19 in the Worldwide', figsize=(40,20), logy=True,                            legend=True)
plt.ylabel('Total  Active Cases')


# In[36]:


world_table = world.reset_index()


# In[37]:


x = world_table[world_table['Country'] == 'France']
big_7 = world_table[world_table['Confirmed'] >= x.iloc[0,1]]


# In[38]:


big_7.style.background_gradient(cmap='viridis')


# # MOST AFFECTED COUNTRY

# In[40]:


axs = big_7.plot('Country', ['Confirmed', 'Deaths', 'Recovered', 'active_confirmed'], kind='barh',                 stacked=True, title='Country most affected by novel covid-19',                 figsize=(20,10.5),colormap='rainbow_r', logx=True, legend=True) 


# # TOTAL CASES DAYWISE

# In[41]:


time_obs = impute_covid.groupby('Date')['Confirmed'].aggregate([np.sum])
time_obs.columns = ['Confirmed']


# In[43]:


time_obs.plot(figsize=(20,20), title='novel COVID-19 in the Worldwide', kind='bar')
plt.ylabel('Total Confirmed observation')


# # us special case

# In[48]:


us = impute_covid[impute_covid['Country'] == 'US']


# In[49]:


chstar_date = us.Date.min()
chend_date = us.Date.max()


# In[50]:


lastus = us[us['Date'] == chend_date]
lastus.head()


# # US report

# In[51]:


print('================ US report ===================================')
print('== Information to {} on novel COVID-19 =========\n'.format(chend_date))
print('Tota confirmed: {}\nTotal Deaths: {}\nTotal Recovered: {}\nTotal active confirmed: {}\n'.format(lastus.Confirmed.sum(), lastus.Deaths.sum(), lastus.Recovered.sum(), lastus.active_confirmed.sum()))
print('==================================================================')


# # US Statewise report

# In[52]:


lastus[['Province/State', 'Confirmed', 'Deaths', 'Recovered', 'active_confirmed']].style.background_gradient(cmap='viridis')


# In[53]:


province = lastus.groupby('Province/State').sum()
province = province.sort_values(by=['Confirmed'], ascending=False)


# In[54]:


province.plot(kind='bar', label='Confirmed',logy=True,figsize=(20,10), stacked=True,              title='US state with novel covid-19')
plt.ylabel('Total patient')


# # confirmed in US daywise

# In[55]:


conf_us = us.groupby('Date')['Confirmed'].agg('sum')
rec_us = us.groupby('Date')['Recovered'].agg('sum')
dea_us = us.groupby('Date')['Deaths'].agg('sum')
ac_us = us.groupby('Date')['active_confirmed'].agg('sum')


# In[56]:


conf_us.plot(figsize=(20,12), kind='bar',title='observationdate of patient confirmed in US')
plt.ylabel('Total patient')


# # rest of world

# In[57]:


rest_world = impute_covid[impute_covid['Country'] != 'US']


# In[58]:


rest_world.head()


# In[59]:


row = rest_world[rest_world['Date'] == rest_world.Date.max()]


# In[60]:


print('================ ROW report =====================================')
print('== Information to {} on novel COVID-19 =========\n'.format(chend_date))
print('Tota confirmed: {}\nTotal Deaths: {}\nTotal Recovered: {}\nTotal active confirmed: {}\n'.format(row.Confirmed.sum(), row.Deaths.sum(), row.Recovered.sum(), row.active_confirmed.sum()))
print('==================================================================')


# In[61]:


rw = row[['Country', 'Confirmed', 'Deaths', 'Recovered', 'active_confirmed']].groupby('Country').sum()
rwx = rw.sort_values(by=['Confirmed'], ascending=False)
rwx.style.background_gradient(cmap='viridis')


# In[ ]:




