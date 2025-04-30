// DISCLAIMER AND DATASETS USED // 

THE YELP DATASET
- Note that I am unable to provide the Yelp datasets from their API updated in 2025 based on the agreement terms to use the Yelp API
- The API is very easy to obtain though with a quick google search
- Use the 'business' and 'review' datasets from the larger API

THE 2023 US CENSUS DATA
- This data is also open-source just like the Yelp API 
- use the following link to acess the dataset: https://www.census.gov/data/tables/time-series/demo/popest/2020s-total-cities-and-towns.html
- this should be labled under 'Vintage 2023' with the subheading 'City and Town Population'

ABOUT THE CODE
- Note that this is original python code created originally in a Jupyter Notebook split into 4 parts, labled accordingly
- Within the python files are documented comments that will explain every step of the way
- Please note the libraries needed to recreate this research project within the python files
- Thank you

// QUICK SUMMARY OF THE RESEARCH PROJECT //

INTRODUCTION
- Explore how Yelp user-review text relates to ratings (1-5), aiming to predict scores using machine learning 
- Process review text through tokenization, removing stopwords, lemmatization, and vectorizing (TF-IDF)
- Introduces novel training approach by incorporating population category: (Major City, Small City, Town)
- 3 Datasets were used:
- Yelp Business Data (Yelp API)
- Yelp Review Data (Yelp API)
- US Government Census Data
- Factoring in population category, the model aims to generalize better across all business types and geographic settings

METHODOLOGY
- Business data had 150,346 records- mapped to review data, totaling 6,990,280 records
- Mapped population numbers from the 2023 census, leaving 1,479,684 records
- Many location names were misspelled, and data cleaning became very difficult- had about 1,000 unique locations
- Percentile-based split (equal thirds) of the population data to ensure evenly distributed training data
- Implemented Random Forest, LightGBM, and Multiclass Logistic Regression

RESULTS 
- After playing with the parameters of multiple models, the multiclass logistic regression had the highest accuracy
- This model was able to predict a Yelp score based on text language with around 69% accuracy
- The scoring system: (1/5: Terrible Experience à 5/5: Incredible Experience (with 2-4 showing mixed sentiments))
- Very effective at predicting 5/5 based on review vocabulary, as well as 1/5 scores
- Much better at predicting scores closer to 5/5 (usually 4s) than predicting 1s, 2s, and 3s accurately
- 5-fold cross-validation accuracy was 0.6857 +/- 0.0016
- Precision: 59.47% / F1- Score: 57.34%
- Multiclass logistic regression performs reasonably well in predicting accurate Yelp score based on text in review

DISCUSSION
- Score rating distribution is heavily skewed toward high ratings (4/5 and 5/5)
- Suggests positive bias, users are more likely to post favorable feedback than negative
- The logistic regression model performed reasonably well with solid accuracy
- Confusion matrix showed that the model struggled with mid-range predictions
- This might be because mid-range scores have similar language in the reviews
- Most reviews came from large population areas, reflecting more businesses and Yelp
activity in an urban environment
- The new ”Population Category” feature adds valuable context to the model but could be
improved
- Shows that Natural Language Processing combined with machine learning can
effectively predict ratings, but results must be interpreted with awareness of data bias
and its limitations

FUTURE WORK / APPLICATION 
- Address missing population data
- Mapping unincorporated areas to the nearest census-
designated places
Handle class imbalance more effectively
- Class weighting or SMOTE to improve performance for
2/5 and 3/5 scores
Additional NLP features
- Use Review Length as a proxy for emotional intensity or
user engagement
- Region-specific language or slang detection
Other Models
- Visualize review language across regions
- Use clustering (like t-SNE or PCA) on the TF-IDF vectors
to find regional review ”styles”
Applications
- Build tools that help new businesses benchmark against
similar ones in similarly populated areas
- Build a “suggestion” tool that would offer a suggested
Yelp score based on language used in reviews to eliminate
bias in user reviews
