INTRODUCTION
Explore how Yelp user-review text relates to ratings (1-5), aiming to predict scores using machine learning 
Process review text through tokenization, removing stopwords, lemmatization, and vectorizing (TF-IDF)
Introduces novel training approach by incorporating population category: (Major City, Small City, Town)
3 Datasets were used:
Yelp Business Data (Yelp API)
Yelp Review Data (Yelp API)
US Government Census Data
Factoring in population category, the model aims to generalize better across all business types and geographic settings

METHODOLOGY
Business data had 150,346 records- mapped to review data, totaling 6,990,280 records
Mapped population numbers from the 2023 census, leaving 1,479,684 records
Many location names were misspelled, and data cleaning became very difficult- had about 1,000 unique locations
Percentile-based split (equal thirds) of the population data to ensure evenly distributed training data
Implemented Random Forest, LightGBM, and Multiclass Logistic Regression
