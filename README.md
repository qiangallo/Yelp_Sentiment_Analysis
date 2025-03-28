# Yelp_Sentiment_Analysis Part 1: Cleaning the Dataset
Exploring how Yelp user-reviews relates to retakings, through using machine learning models to predict score with natural language processing

#Part 1 Cleaning the Dataset -----------------------------------
import pandas as pd
import json

#Review Data from API
with open('/Users/qian/Desktop/DATA_534/Yelp_JSON/yelp_dataset/yelp_academic_dataset_review.json', 'r') as f:
    reviews = [json.loads(line) for line in f]

df_reviews = pd.DataFrame(reviews)
print(df_reviews.head())

#df_reviews.to_csv('yelp_reviews_data.csv', index=False)

#Business Data from API
with open('/Users/qian/Desktop/DATA_534/Yelp_JSON/yelp_dataset/yelp_academic_dataset_business.json', 'r') as f:
    business = [json.loads(line) for line in f]

df_business = pd.DataFrame(business)
print(df_business.head())

#df_business.to_csv('yelp_business_data.csv', index=False)

print(df_business.columns)

print(df_business['city'].unique()[:1000])  # Adjust the number to see more values if needed

#Importing as CSV files for workflow
import pandas as pd
df_reviews = pd.read_csv('/Users/qian/Desktop/DATA_534/yelp_reviews_data.csv')
print(df_reviews.head())

df_business = pd.read_csv('/Users/qian/Desktop/DATA_534/yelp_business_data.csv')
print(df_business.head())

#Merging the datasets on the common 'business_id'
df_merged = pd.merge(df_reviews, df_business, on='business_id', how='inner')
print(df_merged.head())

#Checking to see how many rows I have after the merge, which 
#is about 7 million

print(len(df_merged))

#Saving to a csv file
#df_merged.to_csv('yelp_merged_data.csv', index=False)

#Checking all the city names in the dataset
for city in df_merged['city'].unique():
    print(city)

unique_cities_count = df_merged['city'].nunique()
print(unique_cities_count)

#We see that there are currently 1416 unique city names, but there are duplicate
#cities represented in different ways

df_cleaned = df_merged.copy()  # Create a copy of the original merged DataFrame

#First round of cleaning
def clean_city_name(city):
    """Standardizes city names using regex pattern matching."""
    city = city.strip().title()  # Capitalize properly (e.g., "new york" -> "New York")
    # Use regex to simplify names
    city = re.sub(r'(?i)st[\.\s]*louis', 'St. Louis', city)
    city = re.sub(r'(?i)st[\.\s]*charles', 'St. Charles', city)
    city = re.sub(r'(?i)nashville.*', 'Nashville', city)
    city = re.sub(r'(?i)philadelph.*', 'Philadelphia', city)
    city = re.sub(r'(?i)mount[\.\s]*laurel.*', 'Mount Laurel', city)
    city = re.sub(r'(?i)tampa[\s]*bay', 'Tampa', city)
    city = re.sub(r'(?i)st[\.\s]*pete[\s]*beach', 'St. Pete Beach', city)
    city = re.sub(r'(?i)mt[\.\s]*juliet', 'Mount Juliet', city)
    city = re.sub(r'(?i)tuscon', 'Tucson', city)
    return city

df_cleaned['city'] = df_cleaned['city'].apply(clean_city_name)

print(df_cleaned['city'].unique())

unique_cities_count2 = df_cleaned['city'].nunique()
print(unique_cities_count2)

#Now after a little cleaning there are only 1215 city names
#but I understand now that many cities, might have the same name but located
#in different states
#I focus on this next

for city in df_cleaned['city'].unique():
    print(city)
    
df_cleaned['city_st'] = df_cleaned['city'] + ", " + df_cleaned['state']

print(df_cleaned[['city', 'state', 'city_st']].head())

#Okay, so there are multiple things that still need to be cleaned
#St.Louis, MO, is also written as Saint Louis, MO
#St. Charles, MO for the same reason
#St.Petersburg, FL for the same reason
#St. Pepe Beach. FL, for the same reason
#St. Ann, MO for the same reason
#St. Davids, PA
#St. Albert, AB
#other places have been written different as abbreviations
#Like Mt. Laurel, NJ
#Mt. Ephraim, NJ
#Mt. Holly, NJ
#Mt. Juliet, TN
#Mt. Laurel, NJ vs. Mount Laurel Township, NJ
#Mt. Vernon, NY vs. Mount Vernon, NY
#Mt. Lebanon, PA vs. Mount Lebanon, PA
#Mt. Clemens, MI vs. Mount Clemens, MI
#Others still have spacing, typos, and formatting differences
#Newtown, PA vs. Newtown Square, PA
#Garnet Valley, PA vs. Garnett Valley, PA (possible typo)
#Lansdale, PA vs. Landsdale, PA (possible typo)
#Feasterville Trevose, PA vs. Feasterville-Trevose, PA
#Indianapolis, IN vs. Indianpolis, IN
#Norristown, PA vs. Norristown,PA (missing space)
#West Chester, PA vs. W.Chester, PA (abbreviation)
#And other different, naming conventions
#Largo, FL vs. Largo (Walsingham), FL
#Wesley Chapel, FL vs. Wesley Ccapel, FL
#Land O Lakes, FL vs. Land O' Lakes, FL
#O'Fallon, IL vs. O Fallon, IL
#Upper Darby, PA vs. Upper Darby PA, PA
#Haddon Township, NJ vs. Haddon Twp, NJ
#Bala Cynwyd, PA vs. Cynwyd, PA
#Maple Shade, NJ vs. Maple Shade Township, NJ
#Glen Mills, PA vs. Glenn Mills, PA
#South Pasadena, FL vs. S. Pasadena, FL
#McCordsville, IN vs. Mccordsville, IN

df_2 = df_cleaned.copy() 

city_corrections = {
    "Saint Louis, MO": "St. Louis, MO",
    "Saint Charles, MO": "St. Charles, MO",
    "Saint Petersburg, FL": "St. Petersburg, FL",
    "Saint Pete Beach, FL": "St. Pete Beach, FL",
    "Saint Ann, MO": "St. Ann, MO",
    "Saint Davids, PA": "St. Davids, PA",
    "Saint Albert, AB": "St. Albert, AB",
    "Mount Laurel, NJ": "Mt. Laurel, NJ",
    "Mount Ephraim, NJ": "Mt. Ephraim, NJ",
    "Mount Holly, NJ": "Mt. Holly, NJ",
    "Mount Juliet, TN": "Mt. Juliet, TN",
    "Mount Lebanon, PA": "Mt. Lebanon, PA",
    "Mount Clemens, MI": "Mt. Clemens, MI",
    "Newtown Square, PA": "Newtown, PA",
    "Feasterville-Trevose, PA": "Feasterville Trevose, PA",
    "Land O' Lakes, FL": "Land O Lakes, FL",
    "O Fallon, IL": "O'Fallon, IL",
    "Upper Darby PA, PA": "Upper Darby, PA",
    "Haddon Twp, NJ": "Haddon Township, NJ",
    "Maple Shade Township, NJ": "Maple Shade, NJ",
    "Glen Mills, PA": "Glenn Mills, PA",
    "McCordsville, IN": "Mccordsville, IN",
}
df_2['city_st'] = df_2['city_st'].replace(city_corrections)

unique_cities_count3 = df_2['city_st'].nunique()
print(unique_cities_count3)

if 'city_st' in df_2.columns:
    # Remove extra commas using regex
    df_2['city_st'] = df_2['city_st'].str.replace(r',\s*,', ',', regex=True)
        # Trim any leading or trailing spaces
    df_2['city_st'] = df_2['city_st'].str.strip()
    print(df_2[['city_st']].head())
else:
    print("Column 'city_st' not found in the dataset.")

df_2['city_st'] = df_2['city_st'].str.strip()

unique_cities_count4 = df_2['city_st'].nunique()
print(unique_cities_count4)

df_3 = df_2.copy() 

for city_st in df_3['city_st'].unique():
    print(city_st)

#there are still plently of issues in cleaning the cities

import pandas as pd

# Dictionary mapping incorrect city names to standardized versions
city_standardization = {
    "Saint Louis, MO": "St. Louis, MO",
    "St. Loius, MO": "St. Louis, MO",
    "Saint Petersburg, FL": "St. Petersburg, FL",
    "St Petersburg, FL": "St. Petersburg, FL",
    "Saintt Petersburg, FL": "St. Petersburg, FL",
    "St Charles, MO": "St. Charles, MO",
    "Saint Charles, MO": "St. Charles, MO",
    "Mt. Laurel, NJ": "Mount Laurel, NJ",
    "Mt Laurel, NJ": "Mount Laurel, NJ",
    "Saint Ann, MO": "St. Ann, MO",
    "New Orleans Ap, LA": "New Orleans, LA",
    "Haverford,, PA": "Haverford, PA",
    "New Pt Richey, FL": "New Port Richey, FL",
    "Reno Nevada, NV": "Reno, NV",
    "West Chester Pa, PA": "West Chester, PA",
    "Wesley Ccapel, FL": "Wesley Chapel, FL",
    "Newtown Sqaure, PA": "Newtown Square, PA",
    "Belleair Blufs, FL": "Belleair Bluffs, FL",
    "Saint Rose, LA": "St. Rose, LA",
    "St.Rose, LA": "St. Rose, LA",
    "Land-O-Lakes, FL": "Land O' Lakes, FL",
    "Cheltenham Township, PA": "Cheltenham, PA",
    "Feasterville-Trevose, Pa, PA": "Feasterville Trevose, PA",
    "Ewing Township, NJ": "Ewing, NJ",
    "Ewing Twp, NJ": "Ewing, NJ",
    "Pennsauken Township, NJ": "Pennsauken, NJ",
    "West Deptford Townsh, NJ": "West Deptford, NJ",
    "Cherry Hil, NJ": "Cherry Hill, NJ",
    "Cherry Hill,, NJ": "Cherry Hill, NJ",
    "Deptford Township, NJ": "Deptford, NJ",
    "Deptford Twp, NJ": "Deptford, NJ",
    "Bensalem. Pa, PA": "Bensalem, PA",
    "Tampa,Fl, FL": "Tampa, FL",
    "Tampla, FL": "Tampa, FL",
    "Tampa Florida, FL": "Tampa, FL",
    "Philadephia, PA": "Philadelphia, PA",
    "Philiidelphia, PA": "Philadelphia, PA",
    "Philly, PA": "Philadelphia, PA",
    "Washington Twp, NJ": "Washington Township, NJ",
    "St Albert, AB": "St. Albert, AB",
    "Staint Albert, AB": "St. Albert, AB",
    "Largo (Walsingham), FL": "Largo, FL",
    "Claerwater, FL": "Clearwater, FL",
    "Indianpolis, IN": "Indianapolis, IN",
    "Indianapolis,, IN": "Indianapolis, IN",
    "O Fallon, IL": "O'Fallon, IL",
    "Ofallon, IL": "O'Fallon, IL",
    "New Britian, PA": "New Britain, PA",
    "Upper Darby Pa, PA": "Upper Darby, PA",
    "St Peters, MO": "St. Peters, MO",
    "St Petersberg, FL": "St. Petersburg, FL",
    "St. Loius, MO": "St. Louis, MO",
    "Brentwood - Cool Springs, TN": "Brentwood, TN",
    "Santa Barbra, CA": "Santa Barbara, CA",
    "Santa Barbara,, CA": "Santa Barbara, CA"
}

# Apply the corrections using the dictionary
df_3['city_st'] = df_3['city_st'].replace(city_standardization)

# Display the cleaned city names
print(df_3[['city_st']])

unique_cities_count5 = df_3['city_st'].nunique()
print(unique_cities_count5)

df_4 = df_3.copy() 

for city_st in df_4['city_st'].unique():
    print(city_st)

#still some issues

city_standardization = {
    "Saint Louis, MO": "St. Louis, MO",
    "St Louis, MO": "St. Louis, MO",
    "Saint Petersburg, FL": "St. Petersburg, FL",
    "St Petersburg, FL": "St. Petersburg, FL",
    "Saint Charles, MO": "St. Charles, MO",
    "St Charles, MO": "St. Charles, MO",
    "Saint Ann, MO": "St. Ann, MO",
    "St Ann, MO": "St. Ann, MO",
    "Saint Rose, LA": "St. Rose, LA",
    "St Rose, LA": "St. Rose, LA",
    "Philiadelphia, PA": "Philadelphia, PA",
    "East Norristown, PA": "Norristown, PA",
    "Mount Laurel, NJ": "Mt. Laurel, NJ",
    "Mt Laurel, NJ": "Mt. Laurel, NJ",
    "W. Chester, PA": "West Chester, PA",
    "New Orlaens, LA": "New Orleans, LA",
    "Zephryhills, FL": "Zephyrhills, FL",
    "Land O' Lakes, FL": "Land O Lakes, FL",
    "West Deptford Township, NJ": "West Deptford, NJ",
    "Newcastle, DE": "New Castle, DE",
    "Mount Holly, NJ": "Mt. Holly, NJ",
    "Mt Holly, NJ": "Mt. Holly, NJ",
    "West Berlin, NJ": "W. Berlin, NJ",
    "Upper Darby Pa, PA": "Upper Darby, PA",
    "Ridley Township, PA": "Ridley, PA",
    "Cherry Hil, NJ": "Cherry Hill, NJ"
}

df_4['city_st'] = df_4['city_st'].replace(city_standardization)

print(df_4[['city_st']])

unique_cities_count6 = df_4['city_st'].nunique()
print(unique_cities_count6)

for city_st in df_4['city_st'].unique():
    print(city_st)
  
df_5 = df_4.copy()

import pandas as pd
city_standardization = {
    "Saint Louis, MO": "St. Louis, MO",
    "St Louis, MO": "St. Louis, MO",
    "Saint Petersburg, FL": "St. Petersburg, FL",
    "St Petersburg, FL": "St. Petersburg, FL",
    "Saint Ann, MO": "St. Ann, MO",
    "St Ann, MO": "St. Ann, MO",
    "Saint Rose, LA": "St. Rose, LA",
    "St Rose, LA": "St. Rose, LA",
    "Tampa,, FL": "Tampa, FL",
    "Tampla, FL": "Tampa, FL",
    "Tampa Florida, FL": "Tampa, FL",
    "Feasterville-Trevose, PA": "Feasterville Trevose, PA",
    "Newtown Sqaure, PA": "Newtown Square, PA",
    "Cheltenham Township, PA": "Cheltenham, PA",
    "Gibbsboro, NJ": "Gibsboro, NJ",
    "Metairie, LA": "Metairie, LA",
    "Metarie, LA": "Metairie, LA",
    "Nashville, TN": "Nashville, TN",
    "Nsshville, TN": "Nashville, TN",
    "Tucson, Arizona, AZ": "Tucson, AZ",
    "Tuson, AZ": "Tucson, AZ",
    "Berlin, NJ": "Berlin, NJ",
    "Berlin Township, NJ": "Berlin, NJ",
    "Berlin Boro, NJ": "Berlin, NJ",
    "Mount Laurel, NJ": "Mt. Laurel, NJ",
    "Mt Laurel, NJ": "Mt. Laurel, NJ",
    "Mount Holly, NJ": "Mt. Holly, NJ",
    "Mt Holly, NJ": "Mt. Holly, NJ",
    "West Chester Pa, PA": "West Chester, PA",
    "West Deptford Township, NJ": "West Deptford, NJ",
    "Newcastle, DE": "New Castle, DE",
    "Upper Darby Pa, PA": "Upper Darby, PA",
    "Ridley Township, PA": "Ridley, PA",
    "Cherry Hil, NJ": "Cherry Hill, NJ",
    "Cherry Hill,, NJ": "Cherry Hill, NJ",
    "Land O'Lakes, FL": "Land O Lakes, FL",
    "Moorestown, NJ": "Moorestown, NJ",
    "Moorestown-Lenola, NJ": "Moorestown, NJ",
    "Murfreesboro, TN": "Murfreesboro, TN",
    "Murfeesboro, TN": "Murfreesboro, TN"
}

df_5['city_st'] = df_5['city_st'].replace(city_standardization)

print(df_5[['city_st']])

unique_cities_count7 = df_5['city_st'].nunique()
print(unique_cities_count7)

for city_st in df_5['city_st'].unique():
    print(city_st)

df_5_sorted = df_5.sort_values(by='city_st', ascending=True)

for city_st in df_5_sorted['city_st'].unique():
    print(city_st)

df_6 = df_5.copy()

city_standardization6 = {
    "Abington Township, PA": "Abington, PA",
    "Afton, MO": "St. Affton, MO",
    "Ashland, TN": "Ashland City, TN",
    "Belle Chase, LA": "Belle Chasse, LA",
    "Belleair Beach, FL": "Belleair, FL",
    "Belleair Bluffs, FL": "Belleair, FL",
    "Bellafontaine Neighbors, MO": "Bellafontaine, MO",
    "Bensalem, PA": "Bensalem Township, PA",
    "Bethel, PA": "Bethel Township, PA",
    "Boise (Meridian), ID": "Boise, ID",
    "Boise Ap, ID": "Boise, ID",
    "Boise City, ID": "Boise, ID", 
    "Bordentown, NJ": "Bordentown Township, NJ",
    "Bosie, ID": "Boise, ID",
    "Bradenton Beach, FL": "Bradenton, FL",
    "Buckingham, PA": "Buckingham, PA", 
    "Bucks County, PA": "Bucks, PA", 
    "Burlington, NJ": "Burlington Township, NJ",
    "Carney'S Point, NJ": "Carney's Point, NJ",
    "Carneys Point, NJ": "Carney's Point, NJ",
    "Casa Adobes, AZ": "Casas Adobes, AZ",
    "Catalina Foothills, AZ": "Catalina, AZ",
    "Cedarbrook, NJ": "Cedar Brook, NJ",
    "Center City Philadelphia, PA": "Philadelphia, PA",
    "Cherry Hill Mall, NJ": "Cherry Hill, NJ",
    "Clearwater Beach, FL": "Clearwater, FL",
    "Clearwater/ Countryside, FL": "Clearwater, FL",
    "Conshohoeken, PA": "Conshohocken, PA",
    "Cornwell Hts, PA": "Cornwell Heights, PA",
    "Creve Couer, MO": "Creve Coeur, MO",
    "Delaware County, PA": "Delaware, PA"
}

df_6['city_st'] = df_6['city_st'].replace(city_standardization6)

print(df_6[['city_st']])

unique_cities_count8 = df_6['city_st'].nunique()
print(unique_cities_count8)

df_6_sorted = df_6.sort_values(by='city_st', ascending=True)
for city_st in df_6_sorted['city_st'].unique():
    print(city_st)

df_7 = df_6.copy()

city_standardization7 = {
    "Delran Twp, NJ": "Delran Township, NJ",
    "Delran, NJ": "Delran Township, NJ",
    "Devon-Berwyn, PA": "Devon, PA",
    "Downtown Indianapolis, IN": "Indianapolis, IN",
    "Drexel Hil, PA": "Drexel, PA", 
    "Drexel Hill, PA": "Drexel, PA",
    "Edmonton City Centre, AB": "Edmonton, AB",
    "Erdenheim Pa, PA": "Erdenheim, PA",
    "Evesham, NJ": "Evesham Township, NJ",
    "Evshm Twp, NJ": "Evesham Township, NJ",
    "Fairless, PA": "Fairless Hills, PA",
    "Fairview Hts, IL": "Fairview Heights, IL",
    "Fairview Hts., IL": "Fairview Heights, IL",
    "Feasterville Trev, PA": "Feasterville, PA",
    "Feasterville Trevose, PA": "Feasterville, PA",
    "Festerville, PA": "Featerville, PA",
    "Florence, NJ": "Florence Township, NJ",
    "French Quarter - Cbd, LA": "French Quarter, LA",
    "Glenoldan, PA": "Glenolden, PA",
    "Gloucester City, NJ": "Gloucester, NJ",
    "Gloucester Township, NJ": "Gloucester, NJ",
    "Goodletsville, TN": "Goodlettsville, TN", 
    "Green Valle, AZ": "Green Valley, AZ",
    "Gwynedd Valley, PA": "Gwynedd, PA",
    "Hadden, NJ": "Haddon, NJ",
    "Haddon Heights, NJ": "Haddon, NJ",
    "Haddon Township, NJ": "Haddon, NJ",
    "Hamiltion, NJ": "Hamilton, NJ",
    "Hamilton Township, NJ": "Hamilton, NJ",
    "Havertown, Pa, PA": "Havertown, PA",
    "Hernando Bch, FL": "Hernando Beach, FL"
}
df_7['city_st'] = df_7['city_st'].replace(city_standardization7)

print(df_7[['city_st']])

unique_cities_count9 = df_7['city_st'].nunique()
print(unique_cities_count9)

df_7_sorted = df_7.sort_values(by='city_st', ascending=True)
for city_st in df_7_sorted['city_st'].unique():
    print(city_st)

df_8 = df_7.copy()

city_standardization8 = {
    "Hi Nella, NJ": "Hi-Nella, NJ",
    "Hillsborough County, FL": "Hillsborough, FL",
    "Holland Southampton, PA": "Holland, PA",
    "Huntingdon Valley Pa, PA": "Huntingdon Valley, PA",
    "Huntington Valley, PA": "Huntingdon Valley, PA",
    "Indian Rocks Beach., FL": "Indian Rocks Beach, FL",
    "Indianapolis City (Balance), IN": "Indianapolis, IN",
    "Indianopolis, IN": "Indianapolis, IN",
    "Jefferson Parish, LA" : "Jefferson, LA",
    "Kenneth, FL": "Kenneth City, FL",
    "Kimmiswick, MO": "Kimmswick, MO",
    "Kimmswick, MO": "Kimmswick, MO",
    "King Of Prussi, PA": "King Of Prussia, PA",
    "Kng Of Prusia, PA": "King Of Prussia, PA",
    "Landsale, PA": "Landsdale, PA", 
    "Lansdale, PA": "Landsdale, PA",
    "Lawrence, IN": "Lawrence Township, IN",
    "Lawrence, NJ": "Lawrence Township, NJ",
    "Lower Gwynedd, PA": "Lower Gwynedd Township, PA",
    "Lower Southampton, PA": "Lower Southampton Township, PA",
    "Lutz Fl, FL": "Lutz, FL",
    "Madiera Beach, FL": "Madeira Beach, FL",
    "Mantua, NJ": "Mantua Township, NJ",
    "Maple Shade Nj, NJ": "Mantua, NJ",
    "Maran, AZ": "Marana, AZ",
    "Maryland Height, MO": "Maryland Heights, MO",
    "Maryland Height, Mo, MO": "Maryland Heights, MO",
    "Mc Cordsville, IN": "Mccordsville, IN",
    "Medford Lakes, NJ": "Medford, NJ",
    "Mehville, MO": "Mehlville, MO",
    "Meridan, ID": "Meridian, ID"
}

df_8['city_st'] = df_8['city_st'].replace(city_standardization8)

print(df_8[['city_st']])

unique_cities_count10 = df_8['city_st'].nunique()
print(unique_cities_count10)

df_9 = df_8.copy()

df_9_sorted = df_9.sort_values(by='city_st', ascending=True)
for city_st in df_9_sorted['city_st'].unique():
    print(city_st)

city_standardization9 = {
    "Merion Park, PA": "Merion, PA",
    "Merion Station, PA": "Merion, PA",
    "Meterie, LA": "Metairie, LA",
    "Monroe, NJ": "Monroe Township, NJ",
    "Montgomery, PA": "Montgomery Township, PA",
    "Mt. Juliet, TN": "Mount Juliet, TX",
    "Mt. Laurel, NJ": "Mount Laurel, NJ",
    "Mt Laurel Twp, Nj, NJ": "Mount Laurel, NJ",
    "Mount Laurel, NV": "Mount Laurel, NJ",
    "Mt Laurel Township, NJ": "Mount Laurel, NJ",
    "Mt.Laurel, NJ": "Mount Laurel, NJ",
    "Mt. Juliet, TN": "Mount Juliet, TN",
    "Mount Juliet, TX": "Mount Juliet, TN",
    "Mt Ephraim, NJ": "Mt. Ephraim, NJ",
    "N Redngtn Bch, FL": "N Redington Bch, FL",
    "New Orleans, FL": "New Orleans, LA",
    "Newton Square, PA": "Newton, PA",
    "Newtown Sq, PA": "Newton, PA",
    "Newtown Sq., PA": "Newton, PA",
    "Newtown Square, PA": "Newton, PA",
    "Newtown Township, PA": "Newton, PA",
    "Nolenville, TN": "Nolensville, TN",
    "Norritown, PA": "Norristown, PA",
    "North Coventry, PA": "North Coventry Township, PA",
    "North Redington Bch, FL": "North Redington Beach, FL",
    "O' Fallon, IL": "O Fallon, IL",
    "O'Fallon, IL": "O Fallon, IL",
    "Old Hickory, Tn, TN": "Old Hickory, TN",
    "Oldmans, NJ": "Oldmans Township, NJ",
    "Palm Harbor, Fl, FL": "Palm Harbor, FL",
    "Pass-A-Grille, FL": "Pass-A-Grille Beach, FL",
    "Pennsville, NJ": "Pennsville Township, NJ",
    "Phonixville, PA": "Phoenixville, PA"
}
df_9['city_st'] = df_9['city_st'].replace(city_standardization9)

print(df_9[['city_st']])

df_10 = df_9.copy()

df_10_sorted = df_10.sort_values(by='city_st', ascending=True)
for city_st in df_10_sorted['city_st'].unique():
    print(city_st)

city_standardization10 = {
    "Pinellas Park, FL": "Pinellas, FL",
    "Pittsgrove, NJ": "Pittsgrove Township, NJ",
    "Plainfiled, IN": "Plainfield, IN",
    "Plymouth Mtng, PA": "Plymouth Meeting, PA",
    "Primos-Secane, PA": "Primos, PA",
    "Radnor, PA": "Radnor Township, PA",
    "Redingtn Shor, FL": "Redington Shores, FL",
    "Redingtn Shores, FL": "Redington Shores, FL",
    "Redington Shore, FL": "Redington Shores, FL",
    "Reno Ap, NV": "Reno, NV",
    "Reno City, NV": "Reno, NV",
    "Reno Sparks, NV": "Reno, NV",
    "Ridley, PA": "Ridley Park, PA",
    "Riverview Fl, FL": "Riverview, FL",
    "Royford, PA": "Royersford, PA",
    "S.Pasadena, FL": "S Pasadena, FL",
    "Safety  Harbor, FL": "Safety Harbor, FL",
    "Safety Hatbor, FL": "Safety Harbor, FL",
    "Saint Louis Ap, MO": "Saint Louis, MO",
    "Santa  Barbara, CA": "Santa Barbara, CA",
    "Santa Barbara & Ventura Counties, CA": "Santa Barbara, CA",
    "Santa Barbara Ap, CA": "Santa Barbara, CA",
    "Scott Afb, IL": "Scott Air Force Base, IL",
    "Sherwood Park, AB": "Sherwood, AB",
    "Skippack Village, PA": "Skippack, PA",
    "Southampton, NJ": "Southampton Township, NJ",
    "Southwest Philadelphia, PA": "Philadelphia, PA",
    "Southwest Tampa, FL": "Tampa, FL",
    "Sparks Nv, NV": "Sparks, NV",
    "Spark, NV": "Sparks, NV",
    "Springhill, FL": "Spring Hill, FL",
    "St. Louis County, MO": "St. Louis, IL",
    "St. Louis Downtown, MO": "St. Louis, IL",
    "St. Louis, MO": "St. Louis, IL",
    "St. Petersberg, FL": "St. Petersburg, FL"
}
df_10['city_st'] = df_10['city_st'].replace(city_standardization10)

print(df_10[['city_st']])

df_11 = df_10.copy()

df_11_sorted = df_11.sort_values(by='city_st', ascending=True)
for city_st in df_11_sorted['city_st'].unique():
    print(city_st)

city_standardization11 = {
    "Ab Edmonton, AB": "Edmonton, AB",
    "St.Petersburg, FL": "St. Petersburg, FL",
    "St.Ann, MO": "St. Ann, MO",
    "Sun City Center, FL": "Sun City, FL",
    "Tampa - North, FL": "Tampa, FL",
    "Tampa - South, FL": "Tampa, FL",
    "Tampa Ap, FL": "Tampa, FL",
    "Tampa Palms, FL": "Tampa, FL",
    "Tampa Terrace, FL": "Tampa, FL",
    "Temple Terrace, FL": "Temple Terr, FL",
    "Thonosassa, FL": "Thonotosassa, FL",
    "Tierre Verde, FL": "Tierra Verde, FL",
    "Tinicum, PA": "Tinicum Township, PA",
    "Town & Country, MO": "Town And Country, MO",
    "Town & County, MO": "Town And Country, MO",
    "Town 'N' Country, FL": "Town N Country, FL",
    "Treasure Is, FL": "Treasure Island, FL",
    "Tren, NJ": "Trenton, NJ",
    "Tucson Ap, AZ": "Tucson, AZ", 
    "Tucson, HI": "Tucson, AZ",
    "Twn N Cntry, FL": "Town N Country, FL",
    "Upper Southampton, PA": "Upper Southampton Township, PA",
    "Uppr Blck Edy, PA": "Upper Black Eddy, PA",
    "Voorhees, NJ": "Voorhees Township, NJ",
    "W.Chester, PA": "W. Chester, PA",
    "Warrington, PA": "Warrington Township, PA",
    "Washington, NJ": "Washington Township, NJ",
    "Wayne/Radnor, PA": "Wayne, PA",
    "Webster Grvs, MO": "Webster Groves, MO",
    "Wesley Chapel  Fl, FL": "Wesley Chapel, FL",
    "Westampton, NJ": "Westhampton Township, NJ",
    "Westampton Township, NJ": "Westhampton Township, NJ",
    "Westhampton, NJ": "Westhampton Township, NJ",
    "Westmont - Haddon Towsship, NJ": "Westhampton Township, NJ"  
}
df_11['city_st'] = df_11['city_st'].replace(city_standardization11)

print(df_11[['city_st']])

df_12 = df_11.copy()

df_11_sorted = df_11.sort_values(by='city_st', ascending=True)
for city_st in df_11_sorted['city_st'].unique():
    print(city_st)

city_standardization12 = {
    "Whitehouse, TN": "White House, TN",
    "Willingboro, NJ": "Willingboro Township, NJ",
    "Wilmington Manor, DE": "Wilmington, DE",
    "Winslow, NJ": "Winslow Township, NJ",
    "Woodbury Heights, NJ": "Woodbury, NJ",
    "Woodbury Hts., NJ": "Woodbury, NJ",
    "Woolwich Twp, NJ": "Woolwich Township, NJ",
    "Woolwich Twp., NJ": "Woolwich Township, NJ",
    "Wyncote, PA": "Wycombe, PA",
    "Yardley Boro, PA": "Yardley, PA",
    "Zieglerville, PA": "Zieglersville, PA",
    "Zionsville In, IN": "Zionsville, IN"
}
df_12['city_st'] = df_12['city_st'].replace(city_standardization11)

print(df_12[['city_st']])

df_13 = df_12.copy()

# List of city_st values to remove
cities_to_drop = [
    "Arizona, AZ", "Downtown, LA", "Downtown, AB", "Indiana, IN", "Liverpool, XMS", 
    "Scott Air Force Base, IL", "Ste C, FL", "Tennesse, TN", "Virtual, PA", "Wyndlake Condominium, FL"
]

df_14 = df_13[~df_13['city_st'].isin(cities_to_drop)]

print(df_14)

#drop Arizona, AZ
#drop downtown, LA
#drop downtown, AB
#drop, Indiana, IN
#drop Liverpool, XMS
#drop Scott Air Force Base, IL
#drop Ste C, FL
#drop Tennesse, TN
#drop Virtual, PA
#drop Wyndlake Condominium, FL

df_14_sorted = df_14.sort_values(by='city_st', ascending=True)
for city_st in df_14_sorted['city_st'].unique():
    print(city_st)

#FINALLY, it is clean!
