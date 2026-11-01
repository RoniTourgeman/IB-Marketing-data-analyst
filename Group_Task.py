#!/usr/bin/env python
# coding: utf-8

# # Phase 1: Data Ingestion
# 
# ## Initial Dimensionality Check
# Executing structural discovery commands to record row totals, attribute counts, and display the head observations before applying any screening mutations.

# In[1]:


import pandas as pd
import numpy as np

file_path = 'DATA.csv'
df = pd.read_csv(file_path)

raw_rows, raw_cols = df.shape

print(f"Dataset Dimensionality: {raw_rows} Rows, {raw_cols} Columns")
print("-" * 50)
print("Raw Dataset Preview (First 5 Rows):")
df.head(5)


# # Phase 2: Data Cleaning and Column Filtering
# 
# ## 2.1 Dropping Irrelevant Features
# Removing non-informative and anonymized columns ('AdvertisingPlatform', 'AdvertisingTool') that do not provide variance or analytical value for prediction.
# 
# ## 2.2 Duplicate Record Inspection
# Identifying and eliminating completely identical rows to prevent overfitting and ensure data integrity.

# In[2]:


columns_to_drop = ['AdvertisingPlatform', 'AdvertisingTool']
df_cleaned = df.drop(columns=columns_to_drop)

initial_row_count = df_cleaned.shape[0]
df_cleaned = df_cleaned.drop_duplicates()
final_row_count = df_cleaned.shape[0]
duplicates_removed = initial_row_count - final_row_count

print(f"Columns removed: {columns_to_drop}")
print(f"Duplicate rows removed: {duplicates_removed}")
print(f"New Dataset Dimensionality: {df_cleaned.shape[0]} Rows, {df_cleaned.shape[1]} Columns")
print("-" * 50)
print("Cleaned Dataset Preview (First 5 Rows):")
df_cleaned.head(5)


# # Phase 3: Missing Value Assessment
# 
# ## Null and NA Scanning
# Executing an industry-standard validation check across all remaining 18 features to identify null counts, ensuring structural completeness before feature scaling or splitting.

# In[3]:


missing_summary = df_cleaned.isnull().sum()

print("--- Step 3: Missing Values Per Column ---")
print(missing_summary)
print("-" * 50)
print(f"Total missing values found in dataset: {missing_summary.sum()}")


# # Phase 4: Outlier Detection
# 
# ## Statistical Outlier Scanning (IQR Method)
# Using the Interquartile Range (IQR) method to calculate bounds for key numeric metrics, verifying if any extreme or anomalous data points exist that could skew predictive behavior.

# In[4]:


def count_outliers_iqr(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return ((series < lower_bound) | (series > upper_bound)).sum()

features_to_scan = ['Age', 'Income', 'AdSpend', 'ClickThroughRate', 'ConversionRate']
print("--- Step 4: Outliers Detected Per Feature ---")
for col in features_to_scan:
    outlier_count = count_outliers_iqr(df_cleaned[col])
    print(f"Feature '{col}': {outlier_count} outliers found")
print("-" * 50)
print("Outlier assessment completed. Cleaned dataset template:")
df_cleaned.head(5)


# # Phase 5: Categorical Feature Encoding
# 
# ## Categorical to Numeric Transformation
# Applying Binary Label Encoding for 'Gender' and One-Hot Encoding (Dummy Variables) for multi-class variables ('CampaignChannel', 'CampaignType') to ensure the environment is fully numerically optimized.

# In[5]:


# 1. Binary Encoding for Gender (Female = 0, Male = 1)
df_encoded = df_cleaned.copy()
df_encoded['Gender'] = df_encoded['Gender'].map({'Female': 0, 'Male': 1})

# 2. One-Hot Encoding for CampaignChannel and CampaignType
df_encoded = pd.get_dummies(df_encoded, columns=['CampaignChannel', 'CampaignType'], dtype=int)

print("--- Step 5: Categorical Variables Successfully Encoded ---")
print(f"Encoded Dataset Dimensionality: {df_encoded.shape[0]} Rows, {df_encoded.shape[1]} Columns")
print("-" * 65)
print("Fully Encoded Dataset Preview (First 5 Rows):")
df_encoded.head(5)


# # Phase 6: Continuous Feature Normalization (Min-Max Scaling)
# 
# ## Min-Max Transformation to [0,1] Range
# Applying Min-Max scaling to all continuous numeric attributes. This bound transformation maps all features precisely into a positive [0, 1] interval, eliminating magnitude discrepancies while completely avoiding negative values.

# In[6]:


from sklearn.preprocessing import MinMaxScaler

# 1. Identify all continuous numerical columns to scale
numeric_cols = [
    'Age', 'Income', 'AdSpend', 'ClickThroughRate', 'ConversionRate',
    'WebsiteVisits', 'PagesPerVisit', 'TimeOnSite', 'SocialShares', 
    'EmailOpens', 'EmailClicks', 'PreviousPurchases', 'LoyaltyPoints'
]

# 2. Apply MinMaxScaler
scaler = MinMaxScaler()
df_final = df_encoded.copy()
df_final[numeric_cols] = scaler.fit_transform(df_final[numeric_cols])

print("--- Step 6: Continuous Features Successfully Normalized [0, 1] ---")
print(f"Final Cleaned Dataset Shape: {df_final.shape[0]} Rows, {df_final.shape[1]} Columns")
print("-" * 65)
print("Final Production-Ready Dataset Preview (First 5 Rows):")
df_final.head(5)


# # Phase 7: Exporting the Production-Ready Dataset
# 
# ## Saving Processed Data to CSV
# Exporting the fully cleaned, encoded, and Min-Max scaled DataFrame into a static CSV file (`CLEANED_DATA.csv`) to lock the preprocessing environment for subsequent exploratory analysis and modeling.

# In[7]:


output_filename = 'CLEANED_DATA.csv'
df_final.to_csv(output_filename, index=False)

print("--- Step 7: Production-Ready Dataset Exported ---")
print(f"Success: File saved as '{output_filename}' in your working directory.")
print(f"Exported Dimensions: {df_final.shape[0]} Rows, {df_final.shape[1]} Columns")


# # Phase 8: Data Warehouse (DWH) Star Schema Load
# 
# ## Dimension and Fact Table Isolation
# Loading the final optimized, encoded, and normalized dataset into the conceptualized Star Schema structures. This process builds individual dimension tables and maps them via keys to the central marketing interactions fact table.

# In[8]:


# 1. Dim_Customer_Demographics
dim_customer = df_final[['CustomerID', 'Age', 'Gender', 'Income']].drop_duplicates().reset_index(drop=True)

# 2. Dim_Campaign_Channels (Since channel columns were hot-encoded, we pull the encoded indicators)
channel_cols = [col for col in df_final.columns if 'CampaignChannel_' in col or 'CampaignType_' in col]
dim_channels = df_final[channel_cols].drop_duplicates().reset_index(drop=True)
dim_channels.index.name = 'ChannelKey'
dim_channels = dim_channels.reset_index()

# 3. Dim_Customer_Engagement
dim_engagement = df_final[['CustomerID', 'WebsiteVisits', 'PagesPerVisit', 'TimeOnSite', 'SocialShares', 'EmailOpens', 'EmailClicks']].drop_duplicates().reset_index(drop=True)

# 4. Dim_Purchase_History
dim_history = df_final[['CustomerID', 'PreviousPurchases', 'LoyaltyPoints']].drop_duplicates().reset_index(drop=True)

# 5. Fact_Marketing_Interactions
# We map the ChannelKey back by merging on the hot-encoded indicators
fact_marketing = df_final.merge(dim_channels, on=channel_cols, how='left')
fact_marketing = fact_marketing[['CustomerID', 'ChannelKey', 'AdSpend', 'ClickThroughRate', 'ConversionRate', 'Conversion']]

print("--- Step 8: DWH Star Schema Successfully Loaded ---")
print(f"Dim_Customer_Demographics Shape:   {dim_customer.shape}")
print(f"Dim_Campaign_Channels Shape:       {dim_channels.shape}")
print(f"Dim_Customer_Engagement Shape:     {dim_engagement.shape}")
print(f"Dim_Purchase_History Shape:        {dim_history.shape}")
print(f"Fact_Marketing_Interactions Shape:  {fact_marketing.shape}")
print("-" * 65)
print("Fact Table Preview (First 5 Rows):")
fact_marketing.head(5)


# # Phase 9: Exporting DWH Star Schema for BI Integration
# 
# ## Generating Isolated Relational Entities
# Exporting the newly constructed central Fact table and its four corresponding Dimension tables into individual CSV files. This bridges our Python data preprocessing environment with enterprise business intelligence platforms dbdiagram.io for data modeling.

# In[9]:


# Exporting the 5 DWH tables to separate CSV files for dbdiagram.io
dim_customer.to_csv('Dim_Customer_Demographics.csv', index=False)
dim_channels.to_csv('Dim_Campaign_Channels.csv', index=False)
dim_engagement.to_csv('Dim_Customer_Engagement.csv', index=False)
dim_history.to_csv('Dim_Purchase_History.csv', index=False)
fact_marketing.to_csv('Fact_Marketing_Interactions.csv', index=False)

print("--- Step 9: All 5 DWH Tables Exported Successfully to CSV! ---")
print("Files created: Dim_Customer_Demographics, Dim_Campaign_Channels, Dim_Customer_Engagement, Dim_Purchase_History, Fact_Marketing_Interactions.")


# In[ ]:




