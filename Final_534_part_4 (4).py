#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df= pd.read_csv('/Users/qian/Desktop/DATA_534/FINAL_DATASET_534_MARCH_23.csv')
print(df.head())


# In[3]:


print(len(df))


# In[5]:


df_2 = df.copy()


# In[100]:


#Cleaning the text and also tokenizing

import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess_text_faster(text):
    tokens = word_tokenize(str(text).lower())  # Tokenize & lowercase
    return " ".join([lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words])  # Keep only words

# Process in batches
batch_size = 1000

for i in range(0, len(df), batch_size):
    print(f"Processing rows {i} to {min(i + batch_size, len(df))}...")
    df.loc[i:i+batch_size, "Processed Text"] = df.loc[i:i+batch_size, "text"].apply(preprocess_text_faster)

print("Batch processing complete!")


# In[102]:


#YESSSS The Processed Text Exists!!!
df.head()


# In[15]:


print(len(df))


# In[104]:


df_2 = df.copy()


# In[106]:


#Converting Text to TF-IDF Features
from sklearn.feature_extraction.text import TfidfVectorizer

# Convert text into numerical representation using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)  # Limit vocabulary size for efficiency
X_text = vectorizer.fit_transform(df_2["Processed Text"])

# Check the shape of the TF-IDF matrix
print("TF-IDF Matrix Shape:", X_text.shape)


# In[108]:


df_3 = df_2.copy()


# In[110]:


#Encoding Population Category as Numeric Values


from sklearn.preprocessing import LabelEncoder

# Encode 'Population Category'
label_encoder = LabelEncoder()
df_3["Population Category Encoded"] = label_encoder.fit_transform(df["Population Category"])

# Check encoding
print(df_3[["Population Category", "Population Category Encoded"]].head(10))



# In[112]:


#Normalizing Population (Scaling) this is to prevent large population numbers 
#from skewing the model, make the values between 0 and 1, making the model more stable

from sklearn.preprocessing import MinMaxScaler

# Scale 'Population 2023' between 0 and 1
scaler = MinMaxScaler()
df_3["Population 2023 Scaled"] = scaler.fit_transform(df[["Population 2023"]])

# Check the transformation
print(df_3[["Population 2023", "Population 2023 Scaled"]].head(10))


# In[114]:


df_4 = df_3.copy()


# In[ ]:


#Combining the 3 features I will train the model on 
#TF-IDF Text Features 
#Population Category Encoded 
#Population 2023 Scaled


# In[116]:


import scipy.sparse as sp

# Combine all features into one matrix
X_combined = sp.hstack((
    X_text,  # TF-IDF Features
    df_4["Population Category Encoded"].values.reshape(-1, 1),  # Encoded Population Category
    #df_4["Population 2023 Scaled"].values.reshape(-1, 1)  # Scaled Population 2023
))

# Check final shape
print("Final Feature Shape:", X_combined.shape)


# In[118]:


df_4.head()


# In[120]:


from sklearn.model_selection import train_test_split

# Define target variable (rating/stars)
y = df_4["stars_x"]

# Split dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Check sizes
print(f"Training Set: {X_train.shape}, Testing Set: {X_test.shape}")


# In[122]:


# Split training set into training (80%) and validation (20%)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Check sizes
print(f"Final Training Set: {X_train.shape}")
print(f"Validation Set: {X_val.shape}")
print(f"Testing Set: {X_test.shape}")


# In[154]:


from sklearn.ensemble import RandomForestClassifier

# Initialize the model
rf_model = RandomForestClassifier(n_estimators=200, max_depth = 20, n_jobs =-1, random_state=42)

# Train the model on training data
rf_model.fit(X_train, y_train)

print("training complete!")


# In[156]:


from sklearn.metrics import accuracy_score

# Make predictions on the training set
y_train_pred = rf_model.predict(X_train)

# Calculate training accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)

print(f"Training Accuracy: {train_accuracy:.4f}")


# In[158]:


from sklearn.metrics import accuracy_score, classification_report

# Make predictions on the validation set
y_val_pred = rf_model.predict(X_val)

# Evaluate model performance
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy:.4f}")


# In[160]:


from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Generate confusion matrix
cm = confusion_matrix(y_val, y_val_pred)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=rf_model.classes_, yticklabels=rf_model.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Print classification report
print(classification_report(y_val, y_val_pred))


# In[162]:


from sklearn.model_selection import cross_val_score

# Perform cross-validation
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring="accuracy")

# Print the mean and standard deviation of accuracy across folds
print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

#First Round Cross-Validation Accuracy 0.5363 ± 0.0006 with Random Forest n_estimators=200, max_depth = 20, n_jobs =-1, random_state=42


# In[164]:


#!pip install lightgbm


# In[170]:


#LightGBM Model 


import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report

# Define LightGBM model for multiclass classification
lgbm_model = lgb.LGBMClassifier(n_estimators=100, max_depth=10, learning_rate=0.1, objective="multiclass", num_class=5, random_state=42)

# Train the model
lgbm_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], eval_metric="multi_logloss")



# In[172]:


from sklearn.metrics import accuracy_score

# Predict on training and validation sets
y_train_pred = lgbm_model.predict(X_train)
y_val_pred = lgbm_model.predict(X_val)

# Calculate accuracies
train_accuracy = accuracy_score(y_train, y_train_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")


# In[174]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Predict on validation set
y_val_pred = lgbm_model.predict(X_val)

# Compute confusion matrix
cm = confusion_matrix(y_val, y_val_pred)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix for LightGBM")
plt.show()


# In[180]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Predict on validation set
y_val_pred = lgbm_model.predict(X_val)

# Compute confusion matrix
cm = confusion_matrix(y_val, y_val_pred)

# Define true labels (1.0 - 5.0)
labels = [1.0, 2.0, 3.0, 4.0, 5.0]

# Create confusion matrix display with labels
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap="Blues")

# Customize axis labels
plt.xlabel("Predicted Score")
plt.ylabel("True Score")
plt.title("Confusion Matrix for LightGBM")
plt.show()


# In[176]:


from sklearn.model_selection import cross_val_score

# Define LightGBM model
lgbm_cv = lgb.LGBMClassifier(n_estimators=100, max_depth=10, learning_rate=0.1, objective="multiclass", num_class=5, random_state=42)

# Perform 5-fold cross-validation
cv_scores = cross_val_score(lgbm_cv, X_train, y_train, cv=5, scoring="accuracy")

# Print results
print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

#Cross-Validation Accuracy: 0.6508 ± 0.0011


# In[184]:


#Training Logistic Regression Model 

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Initialize Logistic Regression model
log_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=500, random_state=42)

# Train the model on training data
log_reg.fit(X_train, y_train)


# In[186]:


# Predict on training data
y_train_pred = log_reg.predict(X_train)

# Compute accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy:.4f}")


# In[188]:


# Predict on validation data
y_val_pred = log_reg.predict(X_val)

# Compute accuracy
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy:.4f}")


# In[192]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Predict on validation set
y_val_pred = log_reg.predict(X_val)

# Compute confusion matrix
cm = confusion_matrix(y_val, y_val_pred)

# Define true labels (1.0 - 5.0)
labels = [1.0, 2.0, 3.0, 4.0, 5.0]

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap="Blues")

# Customize labels
plt.xlabel("Predicted Score")
plt.ylabel("True Score")
plt.title("Confusion Matrix for Logistic Regression")
plt.show()


# In[194]:


from sklearn.model_selection import cross_val_score

# Perform 5-fold cross-validation
cv_scores = cross_val_score(log_reg, X_train, y_train, cv=5, scoring="accuracy")

# Print results
print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
#Cross-Validation Accuracy: 0.6857 ± 0.0016


# In[196]:


# Predict on test data
y_test_pred = log_reg.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")


# In[198]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Compute confusion matrix
cm = confusion_matrix(y_test, y_test_pred)

# Define true labels (1.0 - 5.0)
labels = [1.0, 2.0, 3.0, 4.0, 5.0]

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap="Blues")

# Customize labels
plt.xlabel("Predicted Score")
plt.ylabel("True Score")
plt.title("Confusion Matrix for Logistic Regression (Test Set)")
plt.show()


# In[200]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Accuracy
accuracy = accuracy_score(y_test, y_test_pred)

# Precision, Recall, F1 (macro = treats all classes equally)
precision = precision_score(y_test, y_test_pred, average="macro")
recall = recall_score(y_test, y_test_pred, average="macro")
f1 = f1_score(y_test, y_test_pred, average="macro")

# Print all metrics
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")


# In[202]:


import matplotlib.pyplot as plt
import seaborn as sns

# Optional: Set plot style
sns.set(style="whitegrid")

# Plot the distribution of star ratings
plt.figure(figsize=(8, 5))
sns.countplot(x="stars_x", data=df, palette="viridis")

# Add labels and title
plt.title("Distribution of Score Ratings")
plt.xlabel("Score Rating")
plt.ylabel("Number of Reviews")
plt.xticks([0, 1, 2, 3, 4], [1.0, 2.0, 3.0, 4.0, 5.0])  # if needed
plt.tight_layout()
plt.show()


# In[210]:


plt.figure(figsize=(10, 6))
sns.countplot(x="stars_x", hue="Population Category", data=df, palette="Set2")

plt.title("Score Ratings by Population Category")
plt.xlabel("Score Rating")
plt.ylabel("Number of Reviews")
plt.legend(title="Population Category")
plt.savefig("star_ratings_by_population2.jpeg", format="jpeg", dpi=300)
plt.tight_layout()
plt.show()

plt.tight_layout()

