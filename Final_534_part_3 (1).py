#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df_3 = pd.read_csv('/Users/qian/Desktop/DATA_534/Final_Data_With_Census.csv')
print(df_3.head())


# In[ ]:


df_3.head()


# In[3]:


df_4 = df_3.copy()


# In[5]:


# Ensure 'Population 2023' exists in the dataset
if "Population 2023" in df_4.columns:
    # Remove commas and convert to float
    df_4["Population 2023"] = df_4["Population 2023"].astype(str).str.replace(",", "").astype(float)

    # Get min, max, and average population
    min_population = df_4["Population 2023"].min()
    max_population = df_4["Population 2023"].max()
    avg_population = df_4["Population 2023"].mean()

    print("Min Population:", min_population)
    print("Max Population:", max_population)
    print("Average Population:", avg_population)
else:
    print("Column 'Population 2023' not found in the dataset.")


# In[7]:


#Choosing to use Percentile-Based Splitting to enure evenly distributed data for training
df_5 = df_4.copy()


# In[9]:


# Calculate percentiles
small_threshold = df_5["Population 2023"].quantile(0.33)
medium_threshold = df_5["Population 2023"].quantile(0.66)

# Categorize based on percentiles
def categorize_population(pop):
    if pop <= small_threshold:
        return "Small"
    elif pop <= medium_threshold:
        return "Medium"
    else:
        return "Large"

df_5["Population Category"] = df_5["Population 2023"].apply(categorize_population)

# Check category counts
print(df_5["Population Category"].value_counts())


# In[11]:


df_5.head()


# In[15]:


print(df_5.columns)


# In[17]:


df_6 = df_5.copy()


# In[29]:


df_6 = df_6.drop(["user_id", "useful", 'funny', 'cool', 'date', 'address', 'city', 'state',
             'postal_code', 'latitude', 'longitude', 'review_count', 'is_open', 'hours',], axis=1)


# In[31]:


df_6.head()


# In[33]:


df_7 = df_6.copy()


# In[35]:


df_7 = df_7.drop(["stars_y"], axis=1)


# In[37]:


df_7.head()


# In[39]:


df_8 = df_7.copy()


# In[41]:


df_8 = df_8.dropna(subset=["Population 2023"])


# In[43]:


print(len(df_8))


# In[45]:


df_8.head()


# In[48]:


df_8.to_csv("FINAL_DATASET_534_MARCH_23.csv", index=False)


# In[50]:


df_9 = df_8.copy()


# In[54]:


#!pip install spacy


# In[58]:


#!pip install --upgrade spacy pydantic

