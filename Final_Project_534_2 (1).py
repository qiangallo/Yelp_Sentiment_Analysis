#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
df = pd.read_csv('/Users/qian/Desktop/DATA_534/True_Cleaned_Names_Dataset_534.csv')
print(df.head())


# In[4]:


df_sorted= df.sort_values(by='city_st', ascending=True)
for city_st in df['city_st'].unique():
    print(city_st)


# In[6]:


df_census = pd.read_csv('/Users/qian/Desktop/DATA_534/census_data.csv', encoding='latin-1')
df_census.head()


# In[8]:


df_census_2 = df_census.copy()


# In[10]:


# Sample DataFrame
df_census_2 = pd.read_csv('/Users/qian/Desktop/DATA_534/census_data.csv', encoding='latin-1', skiprows=3)  # Skipping the first 3 rows

# Rename columns manually
df_census_2.columns = [
    "Rank", 
    "Geographic Area", 
    "2020 Estimates Base", 
    "Population 2020", 
    "Population 2021", 
    "Population 2022", 
    "Population 2023"
]

# Display the first few rows
print(df_census_2.head())


# In[12]:


df_census_3 = df_census_2.copy()


# In[14]:


# Drop the unwanted columns
df_census_3 = df_census_3.drop(columns=["Population 2020", "Population 2021", "Population 2022", "2020 Estimates Base"])

# Display the first few rows to verify
print(df_census_3.head())


# In[16]:


df_census_4 = df_census_3.copy()


# In[18]:


import pandas as pd

# Dictionary mapping full state names to abbreviations
state_abbreviations = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
    "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH",
    "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
    "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
    "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN",
    "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY"
}

# Function to replace full state names with abbreviations, handling NaN values
def replace_state_names(city_state):
    if isinstance(city_state, str):  # Only process if it's a string
        for full_state, abbrev in state_abbreviations.items():
            if full_state in city_state:
                return city_state.replace(full_state, abbrev)
    return city_state  # Return unchanged if NaN or not a string

# Apply function to the "Geographic Area" column
df_census_4["Geographic Area"] = df_census_4["Geographic Area"].apply(replace_state_names)

# Display the updated DataFrame
print(df_census_4.head())


# In[20]:


df_census_5 = df_census_4.copy()


# In[22]:


# Remove the word "city" (case-insensitive) from Geographic Area
df_census_5["Geographic Area"] = df_census_5["Geographic Area"].str.replace(r"\b[Cc]ity\b", "", regex=True).str.strip()

# Display the updated DataFrame
print(df_census_5.head())


# In[24]:


df_census_6 = df_census_5.copy()


# In[26]:


df_census_6 = df_census_6.rename(columns={"Geographic Area": "city_st"})
df_census_6.head()


# In[28]:


df_census_7 = df_census_6.copy()


# In[30]:


df_census_7["city_st"] = df_census_7["city_st"].str.replace(r"\s+,", ",", regex=True)
df_census_7.head()


# In[32]:


df_census_8 = df_census_7.copy()


# In[34]:


# Create a dictionary mapping city_st to Population 2023 from df_census_6
population_dict = df_census_8.set_index("city_st")["Population 2023"].to_dict()

# Map the population values to df based on city_st
df["Population 2023"] = df["city_st"].map(population_dict)

# Display the updated dataframe
print(df.head())


# In[36]:


print(len(df))


# In[38]:


df.to_csv("Final_Data_With_Census.csv", index=False)


# In[39]:


df.head()


# In[ ]:


#Alright got the clean data set!

