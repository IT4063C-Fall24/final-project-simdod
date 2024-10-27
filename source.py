#!/usr/bin/env python
# coding: utf-8

# # {Project Title}📝
# The Impact of Remote Work on Employee Productivity
# ![Banner](./assets/banner.jpeg)

# In[27]:


#Imports

import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## Topic
# *What problem are you (or your stakeholder) trying to address?*
# 📝 <!-- Answer Below -->
# I want to aim to undertsand how remote work has impacted employees especially after COVID-19. More and more companies are becoming remote so I want to show how these changes to beoming remote impact employees in a good way or bad way. 

# ## Project Question
# *What specific question are you seeking to answer with this project?*
# *This is not the same as the questions you ask to limit the scope of the project.*
# 📝 <!-- Answer Below -->
# 1. How has remote work affected employees personal life? 
# 2. What are some of the benefits of remote work? As well as the negativies?
# 3. How has remote work affected employee productivity?
# 4. How has remote work improved or worsened employee mental health? 

# ## What would an answer look like?
# *What is your hypothesized answer to your question?*
# 📝 <!-- Answer Below -->
# 1. Remote work has affected employees personal lives in a postitive way becuase it allows them to spend more time at homw with their families. 
# 2. Some benefits of remote work are that it allows employees to have a better work life balance. Some negatives are that it can be hard to communicate with coworkers.
# 3. Remote work affects the productivity of employees becuase it allows them to work in a comfortable environment and at their own pace. 
# 4. Remote work has improved employees mental health becuase it has allowed them to be comfortable when working. 

# ## Data Sources
# *What 3 data sources have you identified for this project?*
# *How are you going to relate these datasets?*
# 📝 <!-- Answer Below -->
# Remote work prodcutivity dataset, remote work mental health dataset
# 
# kaggle kernels pull alaaabdelstar/remote-work-productivity
# kaggle kernels pull alaaabdelstar/remote-work-productivity-mental-health

# In[28]:


# Download latest version
path = kagglehub.dataset_download("waqi786/remote-work-and-mental-health")
path1 = kagglehub.dataset_download("mrsimple07/remote-work-productivity")
def load_csv_data():
    file_path = r"C:\Users\simra\OneDrive - University of Cincinnati\UC\Data Tech Analytics\remote-work-productivity_exported.csv"
print("Path to dataset files:", path)
print("Path to dataset files:", path1)


# ## Approach and Analysis
# *What is your approach to answering your project question?*
# *How will you use the identified data to answer your project question?*
# 📝 <!-- Start Discussing the project here; you can add as many code cells as you need -->
# I want to use the databasees to show how productivity has increased over the years due to remote work especially after COVID-19. I also want to show stats about how mental health has increased or decreased becuase of remote work 
# using the data, i can then make a conlcusions about how remote work has improved or worsened over the years and how it has affected employees work and personal life. 
# I want to merge the two datasets of productivity and mental health by using the well being score and the stree level to find out the different levels of stress. 
# 

# Data Cleaning and Tranformation
# - Missing Values
# - Duplicate Values 
# - Anomlies and Outliers 
# - Data Types Transformation 

# In[29]:


# Load data
data = pd.read_csv(r"C:\Users\simra\OneDrive - University of Cincinnati\UC\Data Tech Analytics\remote-work-productivity_exported.csv")
mental_health_data = pd.read_csv(r"C:\Users\simra\.cache\kagglehub\datasets\waqi786\remote-work-and-mental-health\versions\1\Impact_of_Remote_Work_on_Mental_Health.csv")
productivity_data = pd.read_csv(r"C:\Users\simra\.cache\kagglehub\datasets\mrsimple07\remote-work-productivity\versions\1\remote_work_productivity.csv")


# View basic information about the datasets
print("Mental Health Data Overview:")
print(mental_health_data.info())
print(mental_health_data.head())

print("\nProductivity Data Overview:")
print(productivity_data.info())
print(productivity_data.head())
print(data.describe())

# Plot distributions of numeric variables
# Correlation matrix and heatmap for mental health data
numeric_mental_health_data = mental_health_data.select_dtypes(include=['float', 'int'])
mental_corr = numeric_mental_health_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(mental_corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Mental Health Data: Correlation Matrix")
plt.show()

# Correlation matrix and heatmap for productivity data
numeric_productivity_data = productivity_data.select_dtypes(include=['float', 'int'])
prod_corr = numeric_productivity_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(prod_corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Productivity Data: Correlation Matrix")
plt.show()

# Plot boxplots to detect outliers in both datasets
for column in mental_health_data.select_dtypes(include=['float', 'int']):
    sns.boxplot(mental_health_data[column])
    plt.title(f'Mental Health Data: {column}')
    plt.show()

for column in productivity_data.select_dtypes(include=['float', 'int']):
    sns.boxplot(productivity_data[column])
    plt.title(f'Productivity Data: {column}')
    plt.show()

# Plot distributions
for column in data.select_dtypes(include=['float', 'int']):
    sns.histplot(data[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.show()
## detect outliers
for column in data.select_dtypes(include=['float', 'int']):
    sns.boxplot(data[column])
    plt.title(f'Boxplot of {column}')
    plt.show()

 # Check for missing values
print(data.isnull().sum())

print("Missing values in Mental Health Data:")
print(mental_health_data.isnull().sum())

print("\nMissing values in Productivity Data:")
print(productivity_data.isnull().sum())

# Check and remove duplicate rows
duplicates = data.duplicated()
print(f'Total duplicate rows: {duplicates.sum()}')

print(f"Mental Health Data Duplicates: {mental_health_data.duplicated().sum()}")
print(f"Productivity Data Duplicates: {productivity_data.duplicated().sum()}")

mental_health_data = mental_health_data.drop_duplicates()
productivity_data = productivity_data.drop_duplicates()


data_cleaned = data.drop_duplicates()

# Check data types
print(data.dtypes)






# Written explanation of data Cleaning: 
# 
# This cleaning process shows the different outliers in each coloumn along with providing visualizations to help the audience undertsand 
# it better. I also included the missing values, duplicate values and the different datatypes in the cleaning process. From the misisng values 
# you can tell that each coloumn does no have misisng values in the csv file and that there are no duplicates. From the datat types, it is clear
#  that each coloumn has the right data type and is accurate. 
# 
# Written explanation of visualizations: 
# 
# I use the 4 visualizations to showcase the outliers in a different format so that you can see the consisency with it. I used boxplots and
#  histograms as my two libraries. The boxplots were for the outliers, as you can see in the well being store there are many outliers above and
#  below from the average. For productive score there is an outlier below the average of the productivity score. The histograms and bar graphs 
# are used to just see the linear realtionship of the hours work and how it affect the mental acitvity of the employees. I used heatmaps also to 
# find the flow of the mental health of each employee and to show the correlation of each aspect such as hours worked, mental health and 
# productive and how it affects the overall total. 

# Exploratory Data Analysis (EDA)
# 
# - some insights i can find from hese datasets is how each colomun has either a negative or positive impact to productivity and mental health. These visualizations i chose can show each relationship and make it easier to understand the datasets
# - the variables such as hour worked, productivity scores, and mental health are normally distributed and have an either positive or negative realtionship.These variables can help show how each employee is with working at home or at the office and how they have improved from it or decreased their health.  
# - the correlations at this stage are just seeing how each variable can affect the productivity of companies. At this point the relationship between each data is very important to understand what this data is trying to show
# - so far i dont see any data issues or data types that need to be converted 
# 
# 

# Machine Learning Plan:
# 
# We haven't actually worked on the machine learning module yet, that isn't until next week but I hope to use linear regression to predict outcomes from my data. I also want to use clustering to group the eomplyees that are similar together so they can have someone to talk to. 

# ## Resources and References
# *What resources and references have you used for this project?*
# 📝 <!-- Answer Below -->

# In[ ]:


# ⚠️ Make sure you run this cell at the end of your notebook before every submission!
get_ipython().system('jupyter nbconvert --to python source.ipynb')

