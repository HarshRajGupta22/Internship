#!/usr/bin/env python
# coding: utf-8

# # Customer Retention Project

# E-retail factors for customer activation and retention: A case study from Indian e-commerce customers
# 
# Problem Statement:
# 
# Customer satisfaction has emerged as one of the most important factors that guarantee the success of online store; it has been posited as a key stimulant of purchase, repurchase intentions and customer loyalty. A comprehensive review of the literature, theories and models have been carried out to propose the models for customer activation and customer retention. Five major factors that contributed to the success of an e-commerce store have been identified as: service quality, system quality, information quality, trust and net benefit. The research furthermore investigated the factors that influence the online customers repeat purchase intention. The combination of both utilitarian value and hedonistic values are needed to affect the repeat purchase intention (loyalty) positively. The data is collected from the Indian online shoppers. Results indicate the e-retail success factors, which are very much critical for customer satisfaction.
# 
# Since the dataset do not contains target/dependent variable, hence we can consider this as unsupervised learning

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import os
import scipy as stats
from sklearn.preprocessing import LabelEncoder
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')


# In[2]:


# Importing Dataset
# Reading the excel file
df = pd.read_excel("customer_retention_dataset.xlsx")
pd.set_option("display.max_columns",None)
df


# # Observations from the dataset
# 
# 1. The dataset is the combination of both utilitarian value and hedonistic values are needed to affect the repeat purchase intention of the customers.
# 2. Utilitarian values: Utilitarian value is an objective value which provides some functional benefits to the consumers and helps consumers to accomplish practical tasks.
# 3. Hedonistic value: Hedonistic value is subjective (Psychological) value which provides an experiential satisfaction. In other words, the immediate psychological gratification that comes from experiencing some activity or from consumption of a product.
# 4. The dataset contains both numerical,categorical and alphanumerical data.
# 5. The dataset contains the details of of all the customers who shop online frequently and their experienceof buying products. From this details we need to find the success rate of online retailers.

# In[5]:


# To display all the rows in the dataset
pd.set_option("display.max_rows",None)


# # Exploratory Data Analysis (EDA)

# In[5]:


# Checking the dimension of dataset 
df.shape


# The dataset contains 269 rows and 71 columns. The dataset columns have no proper names, so let's rename the names of the columns by appropriate new names.

# In[3]:


# Renaming the column names for better understanding
columns = ['Gender','Age','Shopping_City','Pincode','Shopping_Since','Shopping_Frequency','Internet_Accessibility','Device_Used',
           'Screen_Size','OS', 'Browser_Used','Channel_First_Used','Login_Mode','Time_Explored','Payment_Mode','Abandon_Frequency',
          'Abandon_Reason','Content_Readability','Similar_Product_Info','Seller_Product_Info','Product_Info_Clarity','Navigation_Ease',
          'Loading_Processing_Speed','User_Friendly_Interface','Convenient_Payment_Mode','Timely_Fulfilment_Trust','Customer_Support_Response',
          'Customer_Privacy_Guarantee','Various_Channel_Responses','Benefits','Enjoy','Convenience','Return_Replacement_Policy','Loyalty_Programs_Access',
          'Info_Satisfaction','Site_Quality_Satisfaction','Net_Benefit_Satisfaction','Trust','Product_Several_Category','Relevant_Product_Info','Monetary_Savings',
          'Patronizing_Convenience','Adventure_Sense','Social_Status','Gratification','Role_Fulfilment','Money_Worthy','Shopped_From','Easy_Web_App',
           'Visually_Appealing_WebApp','Product_Variety','Complete_Product_Info','Fast_WebApp','Reliable_WebApp','Quick_Purchase','Payment_Options_Availability',
           'Fast_Delivery','Customer_Privacy_Info','Financial_Security_Info','Perceived_Trustworthiness','Multichannel_Assistance','Long_Login_Time','Long_Display_Time',
           'Late_Price_Declare','Long_Loading_Time','Limited_Payment_Mode','Late_Delivery','WebApp_Design_Change','Page_Disruption','WebApp_Efficiency',
           'Recommendation']

df.columns = columns


# In[8]:


# Checking the new column names after renaming them
df.columns


# # Checking New Name of the Column

# In[8]:


df.head()


# In[9]:


# Checking the type of dataset
df.dtypes


# In[10]:


# To get good overview of the dataset
df.info()


# * This info() method gives the information about the dataset which includes indexing type, column type, no-null values and memory usage.
# * Since counts of all the columns are same, which means there are no null values present in the dataset.

# In[11]:


# Checking number of unique values in each column
df.nunique().to_frame("No of Unique Values")


# In[12]:


# Checking null values in the dataframe
df.isnull().sum()


# In[13]:


# Let's visualize the null values clearly
plt.figure(figsize=(25,10))
sns.heatmap(df.isnull(),cmap="flag")
plt.show()


# From the above heat map, we can clearly notice that there are no null values in any of the columns.
# 
# Let's check the list of value counts in each columns to find if there are any unexpected or corrupted entries present in the dataset

# # Checking the value counts of each columns

# In[15]:


for i in df.columns:
        print(df[i].value_counts())
        print('*'*100)


# These are the list of value counts of each column.
# 
# * In the column Shopping_Frequency, both 41 times and above and 42 times and above belongs to same categories, so will replace them by 41 times and above.
# * The column Internet_Accessibility contains Mobile internet and Mobile Internet which belongs to the same category. So we will replace them using respective categories.
# * Also the column Abandon_Frequency also contains same categories like Frequently and Very frequently, we need to replace them by Frequently.
# * From the column 18-47, we can notice the same type of categories like strongly agree, agree and strongly disagree and disagree.
# 
# We will replace all the same categories by appropriate values.

# In[4]:


# Replacing 42 times and above by 41 times and above in the column Shopping_Frequency
df["Shopping_Frequency"]=df["Shopping_Frequency"].replace('42 times and above','41 times and above')

# Replacing Mobile internet by Mobile Internet in the column Internet_Accessibility
df["Internet_Accessibility"]=df["Internet_Accessibility"].replace('Mobile internet','Mobile Internet')

# Replacing Very frequently by Frequently in the column Abandon_Frequency
df["Abandon_Frequency"]=df["Abandon_Frequency"].replace('Very frequently','Frequently')
# Replacing Strongly agree (5) by Agree (4) in the column Content_Readability
df["Content_Readability"]=df["Content_Readability"].replace('Strongly agree (5)','Agree (4)')

# Replacing Strongly agree (5) by Agree (4) in the column Similar_Product_Info
df["Similar_Product_Info"]=df["Similar_Product_Info"].replace('Strongly agree (5)','Agree (4)')

# Replacing Strongly agree (5) by Agree (4) and strongly disagree by dis-agree in the column Seller_Product_Info
df["Seller_Product_Info"]=df["Seller_Product_Info"].replace('Strongly disagree (1)','Dis-agree (2)')
df["Seller_Product_Info"]=df["Seller_Product_Info"].replace('Strongly agree (5)','Agree (4)')

# Replacing Strongly agree (5) by Agree (4) and Strongly disagree (1) by Dis-agree(2) in the column Product_Info_Clarity
df["Product_Info_Clarity"]=df["Product_Info_Clarity"].replace('Strongly disagree (1)','Dis-agree (2)')
df["Product_Info_Clarity"]=df["Product_Info_Clarity"].replace('Strongly agree (5)','Agree (4)')

# Replacing Strongly disagree (1) by Dis-agree(2) and Strongly agree (5) by Agree (4) in the column Navigation_Ease
df["Navigation_Ease"]=df["Navigation_Ease"].replace('Strongly disagree (1)','Dis-agree (2)')
df["Navigation_Ease"]=df["Navigation_Ease"].replace('Strongly agree (5)','Agree (4)')

# Replacing Strongly disagree (1) by Dis-agree(2) and Strongly agree (5) by Agree (4) in the column Loading_Processing_Speed
df["Loading_Processing_Speed"]=df["Loading_Processing_Speed"].replace('Strongly disagree (1)','Dis-agree (2)')
df["Loading_Processing_Speed"]=df["Loading_Processing_Speed"].replace('Strongly agree (5)','Agree (4)')

# Replacing Strongly disagree (1) by Dis-agree(2) and Strongly agree (5) by Agree (4) in the column User_Friendly_Interface
df["User_Friendly_Interface"]=df["User_Friendly_Interface"].replace('Strongly disagree (1)','Dis-agree (2)')
df["User_Friendly_Interface"]=df["User_Friendly_Interface"].replace('Strongly agree (5)','Agree (4)')

# Replacing Strongly agree (5) by Agree (4) in the column Convenient_Payment_Mode
df["Convenient_Payment_Mode"]=df["Convenient_Payment_Mode"].replace('Strongly agree (5)','Agree (4)')

# Replacing Strongly agree (5) by Agree (4) in the column Timely_Fulfilment_Trust
df["Timely_Fulfilment_Trust"]=df["Timely_Fulfilment_Trust"].replace('Strongly agree (5)','Agree (4)')

# Replacing Strongly agree (5) by Agree (4) in the column Customer_Support_Response
df["Customer_Support_Response"]=df["Customer_Support_Response"].replace('Strongly agree (5)','Agree (4)')

# Replacing Strongly agree (5) by Agree (4) in the column Customer_Privacy_Guarantee
df["Customer_Privacy_Guarantee"]=df["Customer_Privacy_Guarantee"].replace('Strongly agree (5)','Agree (4)')

# Replacing Strongly agree (5) by Agree (4) in the column Various_Channel_Responses
df["Various_Channel_Responses"]=df["Various_Channel_Responses"].replace('Strongly agree (5)','Agree (4)')

# Replacing Strongly disagree (1) by Dis-agree(2) Strongly agree (5) by Agree (4) in the column Benefits
df["Benefits"]=df["Benefits"].replace('Strongly disagree (1)','Dis-agree (2)')
df["Benefits"]=df["Benefits"].replace('Strongly agree (5)','Agree (4)')

# Replacing Strongly disagree (1) by Dis-agree(2) and Strongly agree (5) by Agree (4) in the column Enjoy
df["Enjoy"]=df["Enjoy"].replace('Strongly disagree (1)','Dis-agree (2)')
df["Enjoy"]=df["Enjoy"].replace('Strongly agree (5)','Agree (4)')

# Replacing Strongly agree (5) by Agree (4) in the column Convenience
df["Convenience"]=df["Convenience"].replace('Strongly agree (5)','Agree (4)')

# Replacing Strongly agree (5) by Agree (4) in the column Return_Replacement_Policy
df["Return_Replacement_Policy"]=df["Return_Replacement_Policy"].replace('Strongly agree (5)','Agree (4)')

# Replacing Strongly disagree (1) by Dis-agree(2) and Strongly agree (5) by Agree (4) in the column Loyalty_Programs_Access
df["Loyalty_Programs_Access"]=df["Loyalty_Programs_Access"].replace('Strongly disagree (1)','Dis-agree (2)')
df["Loyalty_Programs_Access"]=df["Loyalty_Programs_Access"].replace('Strongly agree (5)','Agree (4)')

# Replacing Strongly agree (5) by Agree (4) in the column Info_Satisfaction
df["Info_Satisfaction"]=df["Info_Satisfaction"].replace('Strongly agree (5)','Agree (4)')

# Replacing Strongly agree (5) by Agree (4) in the column Site_Quality_Satisfaction
df["Site_Quality_Satisfaction"]=df["Site_Quality_Satisfaction"].replace('Strongly agree (5)','Agree (4)')

# Replacing Strongly agree (5) by Agree (4) in the column Net_Benefit_Satisfaction
df["Net_Benefit_Satisfaction"]=df["Net_Benefit_Satisfaction"].replace('Strongly agree (5)','Agree (4)')

# Replacing Strongly disagree (1) by Dis-agree(2) and Strongly agree (5) by Agree (4) in the column Trust
df["Trust"]=df["Trust"].replace('Strongly disagree (1)','Dis-agree (2)')
df["Trust"]=df["Trust"].replace('Strongly agree (5)','Agree (4)')

# Replacing Strongly agree (5) by Agree (4) in the column Product_Several_Category
df["Product_Several_Category"]=df["Product_Several_Category"].replace('Strongly agree (5)','Agree (4)')

# Replacing Strongly agree (5) by Agree (4) in the column Relevant_Product_Info
df["Relevant_Product_Info"]=df["Relevant_Product_Info"].replace('Strongly agree (5)','Agree (4)')

# Replacing Strongly agree (5) by Agree (4) in the column Monetary_Savings
df["Monetary_Savings"]=df["Monetary_Savings"].replace('Strongly agree (5)','Agree (4)')

# Replacing Strongly agree (5) by Agree (4) in the column Patronizing_Convenience
df["Patronizing_Convenience"]=df["Patronizing_Convenience"].replace('Strongly agree (5)','Agree (4)')

# Replacing Strongly disagree (1) by Dis-agree(2) and Strongly agree (5) by Agree (4) in the column Adventure_Sense
df["Adventure_Sense"]=df["Adventure_Sense"].replace('Strongly disagree (1)','Dis-agree (2)')
df["Adventure_Sense"]=df["Adventure_Sense"].replace('Strongly agree (5)','Agree (4)')

# Replacing Strongly disagree (1) by Dis-agree(2) and Strongly agree (5) by Agree (4) in the column Social_Status
df["Social_Status"]=df["Social_Status"].replace('Strongly disagree (1)','Dis-agree (2)')
df["Social_Status"]=df["Social_Status"].replace('Strongly agree (5)','Agree (4)')

# Replacing Strongly disagree (1) by Dis-agree(2) and Strongly agree (5) by Agree (4) in the column Gratification
df["Gratification"]=df["Gratification"].replace('Strongly disagree (1)','Disagree (2)')
df["Gratification"]=df["Gratification"].replace('Strongly agree (5)','Agree (4)')

# Replacing Strongly disagree (1) by Dis-agree(2) and Strongly agree (5) by Agree (4) in the column Role_Fulfilment
df["Role_Fulfilment"]=df["Role_Fulfilment"].replace('Strongly disagree (1)','Dis-agree (2)')
df["Role_Fulfilment"]=df["Role_Fulfilment"].replace('Strongly agree (5)','Agree (4)')

# Replacing Strongly agree (5) by Agree (4) in the column Money_Worthy
df["Money_Worthy"]=df["Money_Worthy"].replace('Strongly agree (5)','Agree (4)')


# # Checking dataset after replacing the column names and value counts

# In[17]:


df.head()


# # Dataset Description
# # Statistical summary of dataset

# In[18]:


df.describe()


# The describe() method gives the statistical information of only numerical data. In the dataset Pincode is the only column which contains numerical data. Its giving the information of count, mean,standard deviation, min, IQR and max values of the column.

# # Data Visualization
# # Univariate Analysis

# In[5]:


# Let's plot countplot for some of the features

for i in df.columns[:17]:
    
    plt.figure(figsize=(10,4))
    sns.countplot(df[i],palette='magma',saturation=0.75)
    plt.title(i)
    plt.setp(plt.title(i,pad=10),  color='red', style='italic')
    plt.setp(plt.xlabel(i,labelpad=10), size='large', color='k', style='italic')
    plt.setp(plt.ylabel("count",labelpad=10), size='large', color='k', style='italic')
    plt.xticks(rotation=90)
    plt.show()
    print(45*"--")


# # Observations from Countplot :-
# 1. Seeing the countplot we can say that Female are more attracted towards online shopping or may be the data is more focused on Females.
# 2. Based on the age of customers - "Young and Old"(not working class) age people are not much attracted toward shopping(online). While the customers of middle age(working class) are more included in online shopping.
# 3. Based on the cities - cities like "Delhi ,Greater Noida, Noida & Bangalore" are having highest online customers. There may be a lot of conditions responsible for this like fast & easy deliveries, busy schedule of people in big cities etc.
# 4. Based on the time period - There are many customers who are doing online shopping since more than 4 years while every year new customers are being added.
# 5. Based on no.of times online purchase done in last 1 year - a lot of customers are there who had done online purchase upto 10 times in a year.
# 6. Based on the internet used - most of the customers are using "mobile internet" for online shopping. While "Dial-up" is the rarest option.
# 7. Based on the device used - "Smartphone & Laptop" are mostly used for online shopping.
# 8. Based on screen size of phone - "Others & 5.5 inches" are used more frequently. Keeping in mind the above observations, we can assume that others have some larger screen as of laptops.
# 9. Operating system - Customers maily use Windows/windows mobile , followed by Android and IOS/Mac.
# 10. Search engine - "Google Chrome" is the most popular among the search engine for online shopping.
# 11. Channels used - "Search engine" helps the most for channeling the customers towards online store.
# 12. Rather than E-mails(sent by store) or Social Media platform, customers are using again Search Engine, or they direct use the application or use url to go to their prefered store.
# 13. Exploration Time - Most of the customers are taking enough time before making any purchase.
# 14. Payment Option - More Customers are using Credit/Debit cards followed by CoD , E-Wallets.
# 15. Decision of abandoning -There are some  cases in which customers abandon the items because they are getting better alternative offers followed by change in price or promo code not applied.

# In[19]:


# Distribution plot for the column Pincode
sns.distplot(df["Pincode"],color="b",kde_kws={"shade": True},hist=False)
plt.show()


# The data is not normally distributed in this column and there is skewness present in the data, it is almost skewed to right.
# 
# Since all the columns contains categorical data, so we will visualize the data using both pie plots and count plots.

# # PIE PLOT

# In[6]:


# Pie charts for some of the features 

for i in df.columns[17:47]:
    plt.subplots()
    plt.pie(x=df[i].value_counts(),labels=df[i].value_counts().index,data=df,shadow=True, startangle=60,autopct='%1.1f%%',colors=['gold', 'cyan','limegreen', 'magenta', 'crimson',],
               wedgeprops = {'linewidth': 3.8})
    plt.setp(plt.title(i,fontsize=15,color='darkred'),color='maroon',style='italic')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()


# # Observations from PieChart :
# * Conditions where customers are in favour of "Strongly Agreement or Agree" -
# 
# a). The content on the website must be easy to read and understand.
# 
# b). Information on similar product to the one highlighted is importabbt for product comparison.
# 
# c). Complete information on listed seller and product being offered is imporatnt for purchase decision.
# 
# d). All relevant information on listed products must be stated clearly.
# 
# e). Ease of naavigation in website.
# 
# f). Loading and processing speed.
# 
# g). User friendly interface of the website.
# 
# h). Convenient payment methods.
# 
# i). Trust that the online retail store will fulfill its part of the transaction at the stipulated time.
# 
# j). Empathy(readiness to assist wiht queries) towards the custmers.
# 
# k). Being able to guarantee the privacy of the customer.
# 
# l). Responsiveness, availability of several communication channels (email,online rep, twitter, phone etc).
# 
# m). Online shopping gives monetary benefits and discounts.
# 
# n). Enjoyment is derived from shopping online.
# 
# o). Shopping online is convenient and flexible.

# In[20]:


def generate_pie(i):
    plt.figure(figsize=(10,5))
    plt.pie(i.value_counts(), labels=i.value_counts().index, autopct='%1.2f%%',shadow=True,)
    plt.legend(prop={'size':14})
    plt.axis('equal')
    plt.tight_layout()
    return plt.show()

cols1 = ['Gender', 'Age', 'Internet_Accessibility','OS','Channel_First_Used', 'Payment_Mode','Convenient_Payment_Mode','Customer_Privacy_Guarantee','Benefits','Enjoy','Return_Replacement_Policy','Loyalty_Programs_Access','Info_Satisfaction','Site_Quality_Satisfaction','Net_Benefit_Satisfaction','Trust','Monetary_Savings','Patronizing_Convenience','Adventure_Sense','Social_Status','Gratification','Money_Worthy']

plotnumber=1
for j in df[cols1]:
    print(f"Pie plot for the column:", j)
    print(df[j].value_counts())
    generate_pie(df[j])
    print("*"*125)


# # Observations:
# 1. Gender of respondent: The number of gender of respondent for Female customers have high counts compared to Male customers. That is around 67% of female customers shopped online and only 32% of male customers shopped online.
# 2. Age: The count is high for customers whose age is between 31-40 years and they shopped more from the online stores followed by the customers' age between 21-30 years and 41-50 years.
# 3. Internet_Accessibility: 70% of the customers access Mobile Internet to for online purchase and 28% of the customers used WiFi to shop online and only 1% of the customers used Dial-up method to shop online.
# 4. OS: About 45% of the customers' operating system is Windows/windows Mobile and the count is also high for the same followed by the customers having Android OS.
# 5. Channel_First_Used: Around 85.50% of the customers used Search Engine channel to arrive at their favorite online store for the first time.
# 6. Payment_Mode: Most of the customers prefer to pay the bill using Credit/Debit cards and some of the customers prefer cash on delivery and very few of customers use E-wallets payment methods.
# 7. Convenient_Payment_Mode: 88.85% of the customers agreed to the convenient payment mode and only 11% of the customers disagreed to convenient payment mode method.
# 8. Customer_Privacy_Guarantee: Being able to guarantee the privacy of the customer also got 90% agree. That is the customers are concerned about the unauthorized access to their data. Protecting user privacy will enable stores to drive more revenue and gain more customers. Only 9.67% of the customers in neutral state which means they are in confusion whether to agree with this method or not.
# 9. Benefits: About 70% of the customers agreed that the online shopping gives monetary benifita and discounts.
# 10. Enjoy: 53.90% of the customers agreed that they enjoys online shoppings and only 18% of the customers disagreed.
# 11. Return_Replacement_Policy: Around 73% of the customers strongly agree and 19% of the customers agree that the return and replacement policy helps them making purchase decision. It is evident from the fact that the customers actually not liking the products completely, they are just purchasing the products and returning them in case of any dissatisfaction. So it is important for the online shopping websites to make easy return and replacement policy if they want to retain their customers.
# 12. Loyalty_Programs_Access: 66% of the customers agrees that gaining access to loyalty programs is a benefit of shopping online.
# 13. Info_Satisfaction: 79% of the customers agreed that displaying quality information on the website improves satisfaction of customers since they believe that displaying quality information have significant association with customer satisfaction. And remaining 21% of the customers are in neutral situation.
# 14. Site_Quality_Satisfaction: 97% of the customers agreed that they are satisfied while shopping on a good quality website and 3% of the customers disagreed with it.
# 15. Net_Benefit_Satisfaction: About 81% of the customers agreed that the net benefit derived from shopping online can lead to users satisfaction.
# 16. Trust: 88.85% agreed that the customers satisfaction cannot exists without trust. The companies must learn how to manage the customers' trust.
# 17. Monetary_Savings: 82.90% customers agreed to receive monetary savings.The ecommerce company need to know that the best way to sell online is to make the consumer feel that he is saving money doing so. And not just feel, online shopping should result in a lot of saving for the consumer. This saving would automatically get converted into trust and brand equity for the seller. To do this the online companies should offer the best deals and bargains to the consumer through social platforms. If the retailers gives some discounted prices then the customers can make money savings.
# 18. Patronizing_Convenience: 71% of the customers agreed that the Convenience of patronizing the online retailer.
# 19. Adventure_Sense: 57% of the customers agreed that shopping on the website gives the sense of adventure. The adventures in the shopping websites gives positive activity to experience an amplified enjoyment to the customers while shopping on websites.
# 20. Social_Status: Around 39.78% customers agreed that shopping on preferred e-tailer enhances the social status of the customers.
# 21. Gratification: 47.58% of the customers agreed that they felt gratified while shopping on their favourite e-tailer.
# 22. Money_Worthy: Around 86% of the customers agreed that they are getting value for their money while shopping and 14% of the customers thinks either they are wasting money or getting benefit products from their money.

# # Count plots

# In[7]:


def value_count(column):
    counts=len(df[column].value_counts())
    if counts<5:
        plt.figure(figsize=(10,6))
    elif counts<10:
        plt.figure(figsize=(10,8))
        plt.xticks(rotation=90)
    elif counts<20:
        plt.figure(figsize=(25,6))
        plt.xticks(rotation=90)
    else:
        plt.figure(figsize=(20,6))
        plt.xticks(rotation=90)
    sns.countplot(x=column,data=df,palette="Dark2")
    plt.show()
    print("*"*125)
df1=df.iloc[:,[2,4,5,7,8,10,12,13,15,16,17,18,19,20,21,22,23,25,26,28,31,38,39,45,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70]]
for i in df1:
    print(df1[i].value_counts())
    value_count(i)


# # Observations:
# * Most of the customers from the city Delhi, Greater Noida, Noida and Bangalore are used to shop onine and the shopping count is high in these cities.
# * Most of the customers found shopping online for more than 4 years and the count is high for the same followed by the customers shopping for 2-3 years.
# * In last 1 year, most of the customers were purchased online less than 10 times and only few of the customers purchased online 21-30 times.
# * Most of the customers used Smartphone device to access the online shopping and only few customers used Tablet to access the online shopping.
# * The count is high for others mobile screen size followed by 5.5 inches screen size and 5 inches screen size has least count. That means, the customers who have thier mobile screen size other than mentioned inches shopped more online followed by 5.5 inches and the customers having mobile screen size 5 inches shopped very less.
# * Most of the customers used Google chrome to access the website and only few of the customers used Opera and Mozilla Firefox to access the online shopping website.
# * Most of the customers used Search Engine and Via application to reach the online retail store after their first visit and also some customers used Detect URL to reach the online store. Which means these customers have downloaded their most favourite application to reach the online stores easily.
# * Many customers took more than 15 mins before making the purchase decision and some of the customers explored 6-10 mins.
# * Most of the customers abandoned their shopping cart sometimes and some of the customers abandoned their shopping cart frequently.
# * Around 133 customers abandoned their bag due to some better alternative offer and 54 customers abandoned due to promo code not applicable.
# * Around 90% of the customers agreed that the content on the website is easy to read and understand.
# * Around 77% of the customers agreed that the information on similar product to the one highlighted is important for product comparison.
# * About 70% of the customers agreed that complete information on listed seller and product being offered is important for purchase decision.
# * 88.84% of the customers agreed that all relevant information on listed products must be stated clearly and only 11% of the customers disageed with it.
# * 91% of the customers agreed that ease of navigation in website helps them more.
# * Most of the customers agreed that they have no issues with the loading and processing speed.
# * 87% of the customers agreed with user friendly website interface. Creating new user friendly websites will impact on customers to shop more online. By doing this customers don't have to work around much and overall shoping experience would be smooth.
# * 84% of the customers trusted that the online retail store will fulfill its part of the transaction at the stipulated time.
# * The count is high for the customers who agreed the empathy (readiness to assist with queries) towards the customers in the online shopping website is very helpful. If the online shopping companies ready to assist with customers queries then there will be benefit for both company and the customers.
# * The count is high for the customers who agreed that the responsiveness, availability of several communication channels will help them more while shopping online which means if one channel is not available then customers can easily reach out to other channel to fulfill their benifits. So it is important for the companies to provide various channels to communicate with the customers.
# * 83% of the customers agreed that shopping online is convenient and flexible and 12% of the customers are indifferent which means either they are agreed to this or disagreed and only 5% of the customers completely disagreed with it.
# * Most of the customers agreed to offering a wide variety of listed product in several category and the count is high for the same.
# * Around 86% of the customers would like to have provision of complete and relevant product information in the online shopping website.
# * 47% of the customers agreed that shopping on the website helps them fulfilling certain roles and 33% of the customers are in confusion whether to agree or disagree and only 20% of the customers disagrees with it.
# * Most of the people shopped from Amazon.in, Flipkart.com, Paytm.com, Myntra.com, Snapdeal.com companies and they think that it is easy to use website or applicatiion in these companies.
# * Amazon.in and Flipkart.com have high visual appealing web-page layout compared to others.
# * 48% of the customers says that amazon and flipkart shows wide variety of products in their shopping websited compared to other websites. It's important for the companies to show different types of products to gain customers rate.
# * 37% of the customers liked amazon and flipkart in displaying complete and relevant information of the products.
# * Around 51 customers says that Amazon.in is the fast loading website and application and they liked it. About 44 customers liked the web speed of both amazon and paytm followed by amazon ad flipkart.
# * The count is high for amazon followed by amazon and flipkart which means most of the customers liked the reliability of website or application in amazon and flipkart.
# * Most of the customers likes Amazon's quickness to complete the purchase followed by Flipart's and only few of the customers likes Myntra website.
# * In Amazon and flipkart websites there are several payment options available compared to the other shopping websites.
# * Most of the customers liked Amazon's delivery speed followed by flipkart and snapdeal.
# * Most of the customers trusts amazon followed by flipkart in terms of keeping the privacy of their data information.
# * The count is high for the customers who belives that amazon website keeps their finanacial information secrete also the customers trusts flipkart, Myntra, Snapdeal and paytm in terms of keeping thier finanacial information secured.
# * Most of the customers believed that Amazon has perceived trustworthiness comapared to others. Apart from this customers believed that flipkart and Myntra also have perceived trustworthiness.
# * Most of the customers like Amazon inerms of presence of online assistance through multi-channel followed by flipkart,Myntra and snapdeal.
# * Most of the customers agreed that Amazon takes longer time to get logged them in.
# * Customers believes that Amazon and flipkart takes longer time in display the graphics and photos in sales period.
# * Customers says that Myntra and paytm have late declaration of price in promotion/sales period compared to others.
# * Also Myntra and paytm takes longer page loading time.
# * Snapdeal.com has limited mode of payment on most products followed by Amazon.in.
# * In terms of time taken in product delivery Paytm has highest count followed by Snapdeal.com.
# * Most of the customers disliked change in website/Application design on amazon followed by paytm.
# * Most of the customers disliked frequent disruption when moving from one page to another on amazon, Myntra and snapdeal.
# * Most of the customers believes that Amazon and flipkart website is as efficient as before.
# * Most of the customers would like to recommend amazon retailer to a friend followed by flipkart.

# # Bivariate Analysis

# In[22]:


# Comparision between two variables
plt.figure(figsize=[18,13])

plt.subplot(2,2,1)
plt.title('Checking how long the customers shopping online on the basis of Gender')
sns.countplot(df['Shopping_Since'],hue=df['Gender'],palette="gnuplot");

plt.subplot(2,2,2)
plt.title('Checking which device used most to access the online shopping on the basis of Age')
sns.countplot(df['Device_Used'],hue=df['Age'],palette="hls");

plt.subplot(2,2,3)
plt.title('How the customers access online shopping & how many times they made purchase in 1 year')
sns.countplot(df['Internet_Accessibility'],hue=df['Shopping_Frequency'],palette="gnuplot2");

plt.subplot(2,2,4)
plt.title('In 1 year how many times customers made shopping & which city they shopped more')
sns.countplot(df['Shopping_Frequency'],hue=df['Shopping_City']);

plt.xticks(rotation=90)
plt.subplot(2,2,3).legend(loc ="upper left",title="Shopping_Frequency");
plt.subplot(2,2,4).legend(loc ="upper right",title="Shopping_City");
plt.show()


# # Observations:
# Most of the female customers shopped online from more than 4 years and the count is also high for the females who shopped from 2-3 years. And only few male customers shop online more than 4 years. Which means the female customers are more enthusiastic to buy products from the online shopping websites. Many customers whose age between 31-40 years and 21-30 years used Smartphones followed by Laptops to access the online shopping websites. Most of the customers access the shopping websites more than 31-40 times in 1 year through Mobile Internet to shop the products also most of the customers who used mobile internet to access the online shopping website made online purchase less than 10 times in a year. And only few of the customers used WiFi network to access the shopping store. Most of the customers used ecommerce websites less than 10 times in a year from the city Delhi to shop the products.

# # Comparision between two variables

# In[23]:


plt.figure(figsize=[18,13])

plt.subplot(2,2,1)
plt.title('Comparing screen size and the channel used to arrive at the online store',fontsize=12)
sns.countplot(df['Screen_Size'],hue=df['Channel_First_Used'],palette="Dark2");

plt.subplot(2,2,2)
plt.title('Comparing device used and how many times customers shopped in 1 year',fontsize=12)
sns.countplot(df['Device_Used'],hue=df['Shopping_Frequency'],palette="Set2_r");

plt.subplot(2,2,3)
plt.title('Which OS and browser used to access the ecommerce website',fontsize=12)
sns.countplot(df['OS'],hue=df['Browser_Used'],palette="cividis");

plt.subplot(2,2,4)
plt.title('How frequently the customers abandoned the shopping bags and why?',fontsize=12)
sns.countplot(df['Abandon_Frequency'],hue=df['Abandon_Reason'],palette="mako");

plt.subplot(2,2,2).legend(loc ="upper left",title="Shopping_Frequency");
plt.subplot(2,2,4).legend(loc ="upper right",title="Abandon_Reason");
plt.show()


# # Observations:
# * The customers having their mobile screen size say 6 inches(others) have followed search engine channel to arrive at their favorite online store for the first time. Also the customers who have their screen size 5.5 inches also used search engine channel to access the online shopping store.
# * Most of the customers used Smartphones 31-40 times in an year to access the ecommerce websites to shop the products.
# * Many customers having windows operating system in their device ran Google chrome to access the ecommerce shopping websites and some of the customers having IOS/Mac operating system used Google chrome as well as Safari to reach the online shopping store.
# * Due to Lack of trust on the ecommerce websites, sometimes most of the customers abandoned the websites and some of the customers abandoned the shopping website due to the promo code not applicable. which means, if the product is having the special price or some catalogue price rule is applicable on it.Then coupon code should not be applicable on the products.
# 
# 
# So it is important for the ecommerce companies to create discount price, offers, coupon codes to retain the customers.

# In[24]:


# Comparision between two variables
plt.figure(figsize=[18,13])

plt.subplot(2,2,1)
plt.title('How and which channel used to arrive at the online store for 1st time',fontsize=15)
sns.countplot('Login_Mode',hue='Channel_First_Used',palette="cubehelix",data=df);

plt.subplot(2,2,2)
plt.title('Comparing Time_Explored and Content_Readability of the customers',fontsize=15)
sns.countplot('Time_Explored',hue='Content_Readability',palette="magma",data=df);

plt.subplot(2,2,3)
plt.title('Which payment mode and browser used to pay bill',fontsize=15)
sns.countplot('Payment_Mode',hue='Browser_Used',palette="gist_earth",data=df);

plt.subplot(2,2,4)
plt.title('How frequently the customers abandoned the shopping bags while paying the bill?',fontsize=15)
sns.countplot('Abandon_Frequency',hue='Payment_Mode',palette="spring_r",data=df);

plt.show()


# # Observations:
# * Search engine is the most used channel by the customers to arrive their favourite store for the first time and after visit the website for the first time, most of them used the same channel to reach the online retail store to reshopping the products.
# * Most of the customers agreed that the content on the website is easy to read and understand also they explored more than 15 mins before making the purchase decision and some of the customers strongly disagreed that the content is not good and they explored 6-10 mins before making the purchase decision. So ecommerce websites should enable some images and it should contain clear structure, so that the customers can easily read and understand the content of the product.
# * Most of the customers used google chrome to reach the websites and they preferred to pay their product price using Credit/Debit cards and only few of the customers used Safari browser to reach the e-retail websites.
# * Sometimes the customers used to abandon their selected items and wants to leave without making payment and most of them making the payment using E-wallets methods.

# In[25]:


# Comparision between two variables
plt.figure(figsize=[18,13])

plt.subplot(2,2,1)
plt.title('Comparing Similar_Product_Info and Seller_Product_Info',fontsize=15)
sns.countplot(df['Similar_Product_Info'],hue=df['Seller_Product_Info'],palette="ch:.25");

plt.subplot(2,2,2)
plt.title('Comparing Product_Info_Clarity and Navigation_Ease',fontsize=15)
sns.countplot(df['Product_Info_Clarity'],hue=df['Navigation_Ease'],palette="bright");

plt.subplot(2,2,3)
plt.title('Comparing Loading_Processing_Speed and User_Friendly_Interface',fontsize=15)
sns.countplot(df['Loading_Processing_Speed'],hue=df['User_Friendly_Interface'],palette="tab20b_r");

plt.subplot(2,2,4)
plt.title('Comparing Convenient_Payment_Mode and Timely_Fulfilment_Trust',fontsize=15)
sns.countplot(df['Convenient_Payment_Mode'],hue=df['Timely_Fulfilment_Trust'],palette="ocean");

plt.subplot(2,2,4).legend(loc ="upper left",title="Timely_Fulfilment_Trust");
plt.show()


# # Observations:
# * Most of the customers agreed that the information on similar product to the one highlighted is important for product comparison and also Complete information on listed seller and product being offered is important for purchase decision. In order to buy a product, the ecommerce website must give the complete information about the product and seller information then only the customers can compare the product costs and its details in different websites and they tend to buy that particular product in a particular website.
# * Around 90% of the customers agreed that they should be able to navigate the website easily and the products information in the website must be clearly stated their uses, lifetime, benefits etc.Then only more customers tend to buy those products and can shop easily.
# * Most of the customers agreed with the user friendly interface of the websites which can be easily loaded and processed also these websites' loading and processing capacity is very fast so that the customers like to shop in ecommerce websites. If these websites do not have this much of loading and processing speed then customers don't want to buy the products in this website and they tend to other websites or other options rather than this.
# * Most of the customers agree to the trust that the online retail stores will fulfil its part of the transaction at the stipulated time also most of them very happy with the convenient payment modes given by the websites. In other words, the websites must provide all the possible ways of payment methods then only the customers shop frequently all the time the mode of the payment for customers may not possible sometimes they may choose cash on delivery. So if the retailers provides all type of payment methods then the customers can easily make the payment also it enhances the sales of the ecommerce sites. And the transaction must also be given with some stipulated time otherwise the payments may be failed so they've to provide minimum amount of time which need to be fixed for all.

# In[26]:


# Comparision between two variables
plt.figure(figsize=[18,13])

plt.subplot(2,2,1)
plt.title('Comparing Customer_Support_Response and Customer_Privacy_Guarantee',fontsize=12)
sns.countplot(df['Customer_Support_Response'],hue=df['Customer_Privacy_Guarantee'],palette="bone_r");

plt.subplot(2,2,2)
plt.title('Comparing Various_Channel_Responses and monetary benefits of the customers',fontsize=12)
sns.countplot(df['Various_Channel_Responses'],hue=df['Benefits'],palette="brg");

plt.subplot(2,2,3)
plt.title('Comparing Enjoy and Convenience',fontsize=12)
sns.countplot(df['Enjoy'],hue=df['Convenience'],palette="YlOrRd");

plt.subplot(2,2,4)
plt.title('Comparing Return_Replacement_Policy and Loyalty_Programs_Access',fontsize=12)
sns.countplot(df['Return_Replacement_Policy'],hue=df['Loyalty_Programs_Access'],palette="Accent");

plt.subplot(2,2,2).legend(loc ="upper right",title="Benefits");
plt.subplot(2,2,4).legend(loc ="upper left",title="Loyalty_Programs_Access");

plt.show()


# # Observations:
# * Almost all the customers agreed that ecommerce websites have empathy towards them and these sites being able to guarantee the privacy of the customers. That is the online retailers must be able to resolve all the queries of the customers and they have to assure the customers keeping all their credential secured and should not share with others. If the websites give guarantee about the privacy, then the customers make shopping regularly which will enhance the companies sales.
# * Most of the customers agreed that the online shopping gives monetary benefits and responsiveness, availability of several communication channels will help them more while shopping online which means if one channel is not available then customers can easily reach out to other channel to fulfil their benefits. So, it is important for the online e-tailer companies to provide various channels to communicate with the customers. The ecommerce websites should ask the feedback regarding their services, ratings of the products, reviews etc and also they try to communicate with the customers in different social platform then only customers get satisfied by the e-tailers sites and make more shopping on the particular websites regularly which intends to increase the sales of the company. If one website gives less price and more discount for particular product then the customers tend to shop more in that particular website. So, the companies must try to give less price then customers like their offers and retention also increases.
# * Most of the customers believed that they enjoy online shopping also shopping online is convenient and flexible and some of the customers who disagreed with the enjoyment of the shopping, they are not convenient with the online shopping. Some customers shops online for their enjoyment purpose they are termed to be hedonistic, for them shopping online gives experiential satisfaction. They contribute much for the ecommerce companies by buying all the costly products randomly.
# * Most of the customers agreed that return and replacement policy of the e-tailer is important for purchase decision also gaining access to loyalty programs is a benefit of shopping online. Many return policies have conditional agreements, such as time limits, that must be clearly defined and expressed at the time of purchase or else the customers won't get the chance to return their damaged or dissatisfied products due to this they may not access the same website if they want to shop again. It is evident from the fact that the customers actually not liking the products completely, they are just purchasing the products and returning them in case of any dissatisfaction. So it is important for the online shopping websites to make easy return and replacement policy if they want to retain their customers. Also, by gaining access to loyalty programs, the customers get more and more rewards, increasing their engagement rate and thus bringing more profit to both company and customer.

# In[27]:


# Comparision between two variables
plt.figure(figsize=[18,13])

plt.subplot(2,2,1)
plt.title('How the customers satisfaction improves by displaying the information',fontsize=12)
sns.countplot(df['Info_Satisfaction'],hue=df['Site_Quality_Satisfaction'],palette="bwr_r");

plt.subplot(2,2,2)
plt.title('How the customers trust and sataisfied with the net benifits of the websites',fontsize=12)
sns.countplot(df['Net_Benefit_Satisfaction'],hue=df['Trust'],palette="brg");

plt.subplot(2,2,3)
plt.title('How offering several categories and giving product information impact on the customers',fontsize=12)
sns.countplot(df['Product_Several_Category'],hue=df['Relevant_Product_Info'],palette="inferno");

plt.subplot(2,2,4)
plt.title('Comparing Return_Replacement_Policy and Loyalty_Programs_Access',fontsize=12)
sns.countplot(df['Monetary_Savings'],hue=df['Patronizing_Convenience'],palette="afmhot");

plt.subplot(2,2,2).legend(loc ="upper right",title="Trust");
plt.show()


# # Observations:
# * Many customers agreed that displaying quality information on the website improves satisfaction of customers since they believe that displaying quality information have significant association with customer satisfaction and they are satisfied and happy while shopping on good quality websites. In order to obtain high levels of customer satisfaction, high service quality is needed, which often leads to favourable behavioural intentions also a website with good system quality, information quality, and electronic service quality is a key to success in e-commerce. So, the online e-tailers must display all the information about the product then only customers get an idea to buy the products regularly.
# * Most of the customers agreed that net Benefit derived from shopping online can lead to users’ satisfaction also they believe that user satisfaction cannot exist without trust. The e-tailer should provide crediting points (net benefits) so that the customers tend to buy frequently in order to gain points. Trust is also a major factor for customers to decide whether to buy products from online stores or not also trust helps reduce uncertainty when the degree of familiarity between the customer and transaction security mechanism is insufficient. If customers have a high level of trust toward the website, it is more likely for them to have intention to purchase so it’s important for the ecommerce website to make the customers get trust on them.
# * The customers are more likely to purchase on the same websites if that website offers them a wide variety of products in several category and giving relevant information about the products. Having multiple product lines may allow to grow the ecommerce business and finding accurate and up-to-date information of the product must be stated clearly in the website so that the customers can buy the products without any confusion.
# * In this digital and competitive world, everyone wants to save money, the ecommerce company need to know that the best way to sell online is to make the consumer feel that he is saving money doing so. And not just feel, online shopping should result in a lot of saving for the consumer. This saving would automatically get converted into trust and brand equity for the seller. To do this the online companies should offer the best deals and bargains to the consumer through social platforms. If the retailers give some discounted prices then the customers can make money savings and they tend to purchase in the same websites regularly. Convenience is the important thing for ecommerce and most of the customers agreed with it.

# In[31]:


# Comparision between two variables
plt.figure(figsize=[18,13])

plt.subplot(2,2,1)
plt.title('Which app is easy to use and from which website the customers shopped',fontsize=13)
sns.countplot(x='Shopped_From',hue='Easy_Web_App',palette="crest_r",data=df);
plt.xticks(rotation=90)

plt.subplot(2,2,2)
plt.title('Which app provides attractive products and which webapp offering variety of products',fontsize=13)
sns.countplot(x='Visually_Appealing_WebApp',hue='Product_Variety',palette="copper_r",data=df);
plt.xticks(rotation=90)

plt.subplot(2,2,1).legend(loc='upper right',title="Easy_Web_App");
plt.subplot(2,2,2).legend(loc ="upper left",title="Product_Variety");
plt.show()


# # Observations:
# * There are many websites for selling the products among them Amazon.in, Flipkart.com, Paytm.com, Myntra.com, Snapdeal.com are easy to use and shop. Most of the customers used these websites more, this is because, these websites may provide less price products, good discounts and may have lots of varieties of similar products with different brand.
# * Amazon and Flipkart have high visual appealing web-page layout compared to others that means these websites provides some colourful graphics on the homepage. The more people find the website attractive, there are higher chances that they will stay a little longer in that website, also these websites provide wild variety of products in an attractive manner which makes the customers to buy the product.

# In[39]:


# Comparision between two variables
plt.figure(figsize=[18,13])

plt.subplot(2,2,1)
plt.title('Which app changes the design and disruption when moving from one page to another')
sns.countplot(x='WebApp_Design_Change',hue='Page_Disruption',palette="hot",data=df);
plt.xticks(rotation=90)

plt.subplot(2,2,2)
plt.title('Which app is efficient and which Indian online retailer would I recommend to a friend')
sns.countplot(x='WebApp_Efficiency',hue='Recommendation',palette="Wistia",data=df);
plt.xticks(rotation=90)

plt.subplot(2,2,1).legend(loc='upper right',title="Page_Disruption");
plt.subplot(2,2,2).legend(loc ="upper right",title="Recommendation");


# # Observation
# 
# * Amazon is the website where they frequently change their application designs in order to attract the customers and satisfies the customers’ needs and they tend to make customers by updating everyday as per the trend. But the disadvantages of this website are when moving from one page to other it slows down and sometimes it may shutdown.
# * Amazon is the website which is more efficient as before and I suggest Amazon.com and Flipkart as a best Indian online retailer store for purchasing all types of products, as they provide enormous amounts of benefits.
# 
# 
# Since we have observed all the columns contains object type data, so we need to convert them into numerical by using appropriate encding techniques. Here I am using label encoding method to convert the data.

# # Comparing how the shopping on e-tailer gives sense of adventure and enhances the social status of the customers

# In[28]:


sns.factorplot(x='Adventure_Sense',col='Social_Status',data=df,palette="gist_earth",kind="count")
plt.show()


# # Observations
# 
# * Most of the customers agreed that shopping on the website gives the sense of adventure. The adventures in the shopping websites gives positive activity to experience an amplified enjoyment to the customers while shopping on websites. They also believe that shopping on preferred e-tailer enhances the social status of the customers. Many customers think they are adventuring while shopping online as they search for low cost and high discount products to buy and prefer the same to the others. In this way they think that shopping in the website gives them the adventure.
# * Shopping online won't affect anyone's status and the customers agreed that shopping on preferred e-tailer enhances their social status.

# # Comparing how the shopping on favorite e-tailer makes customer feel gratification and helps them fulfill their certain role

# In[29]:


sns.factorplot(x='Gratification',col='Role_Fulfilment',data=df,palette="spring",kind="count")
plt.show()


# # Observation
# 
# * Most of the customers agreed that they felt gratified while shopping on their favourite e-tailer. This is because the e-tailer companies can successfully make up for a mistake or a dissatisfied customer is to be equally expedient in addressing the customer’s needs.
# * Also, most of the customers agreed that shopping on online website helps them fulfil their certain roles. Fulfilment refers to activities that ensure customers receive what they ordered, including the time of delivery, order accuracy, and delivery condition, also the customers cannot see the product directly before they purchase it. Companies must ensure delivery timeliness, order accuracy, and delivery conditions to provide superior service quality for customers. The companies must understand that the customer satisfaction is an indication of the customer's belief of the probability of a service leading to a positive feeling. If the companies give positive vibration to the customers by giving chance to fulfil their roles then they shop more on that particular website.

# # Checking whether the customers getting value for money spent while shopping online after getting information about the product

# In[30]:


sns.factorplot(x='Money_Worthy',col='Info_Satisfaction',data=df,palette="viridis_r",kind="count")
plt.show()


# # Observation
# 
# * The customers should satisfy with their product that they shopped from the online store then only they agreed that they got value for the money they spent. The companies should display the quality information about the products so that the customers being able to purchase their product and thinks that it worth for money and this comes under utilitarian value.

# # Checking Reliability of the website and quickness to complete purchase

# In[32]:


plt.figure(figsize=(13,6))
plt.title('Checking Reliability of the website and quickness tocomplete purchase',fontsize=15)
sns.countplot(x='Reliable_WebApp',hue='Quick_Purchase',palette="afmhot",data=df);
plt.xticks(rotation=90)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),title="Quick_Purchase")
plt.show()


# # Observation
# 
# * The consumer determines the shop’s reliability based on the information transmitted by the shop and certain sites offer customers the opportunity to purchase items that are used which means they are likely to be the most reliable. Some of the customers completes their purchase very quickly due to the discount, less price, free delivery charges etc provided by the ecommerce websites.
# * From the plot we can notice amazon site is more reliable and most of the customers complete their purchase on amazon very quickly.

# # Checking which website delivers the order soon and what payment mode they use

# In[33]:


plt.figure(figsize=(13,6))
plt.title('Checking which website delivers the order soon and what payment mode they use',fontsize=15)
sns.countplot(x='Payment_Options_Availability',hue='Fast_Delivery',palette="icefire_r",data=df);
plt.xticks(rotation=90)
plt.legend(loc ="upper right",title="Fast_Delivery")
plt.show()


# # Observation 
# * Having different types of payment methods will helps the customers to pay the invoice easily using their choice of payment and if the websites have speedy delivery methods without delivery charge, then the customers like to buy the products in those websites.
# * Here amazon and flip kart have several payment options and amazon indeed has speedy order delivery compared to other websites.

# # Comparing privacy of customers information and perceived trustworthiness

# In[34]:


plt.figure(figsize=(13,6))
plt.title('Comparing privacy of customers information and perceived trustworthiness',fontsize=15)
sns.countplot(x='Customer_Privacy_Info',hue='Perceived_Trustworthiness',palette="winter_r",data=df);
plt.xticks(rotation=90)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),title="Perceived_Trustworthiness")
plt.show()


# # Observation
# 
# * Security/privacy refers to the security of credit card payments and privacy of shared information like name of the customer, address and phone number. Customers are always concerned whether the website would protect them against fraud after a transaction. So, website security and privacy are important to assess the service quality of online stores. The customers think that buying online means taking risk, in this case trust is more important thing for both merchant and customer.
# * Most of the customers trusts amazon followed by flip kart in terms of keeping their privacy of data information secured and the customers who believes that amazon website keeps their financial information as secrete also trusts flip kart, Myntra, Snapdeal and Paytm in terms of keeping their financial information secured. Most of the customers believed that Amazon has perceived trustworthiness compared to others. Apart from this, customers believed that flip kart and Myntra also have perceived trustworthiness.

# # Comparing privacy of customers information and perceived trustworthiness

# In[35]:


plt.figure(figsize=(13,6))
plt.title('Checking which WebAapp keeps the financial info of consumers and presence of online assistance through multi-channel',fontsize=15)
sns.countplot(x='Financial_Security_Info',hue='Multichannel_Assistance',palette="twilight_shifted",data=df);
plt.xticks(rotation=90)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),title="Multichannel_Assistance")
plt.show()


# # Observation
# 
# * The customers trusts that amazon and flip kart keeps their financial information private and they never share any type of information to others.
# * Multi-channel retailing provides several benefits which includes several shoppers like the convenience that is provided through online channels in comparison to physical stores. Most of the customers like Amazon in terms of presence of online assistance through multi-channel.

# # # Comparing Long_Login_Time and Long_Display_Time

# In[36]:


plt.figure(figsize=(13,6))
plt.title('Checking which WebAapp takes longer time to get logged in and displaying graphics and photos',fontsize=15)
sns.countplot(x='Long_Login_Time',hue='Long_Display_Time',palette="hsv",data=df);
plt.xticks(rotation=90)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),title="Long_Display_Time")
plt.show()


# # Observations
# 
# * The customers mostly choose amazon website for buying products as it gives promotions and sales periods in some days, on these days most of the customers attracted by the offers provided by the websites, wants to buy the products. So, amazon will take more time to allow the customers to get login into the site.
# * When there is promotion or sales period, amazon and Myntra takes longer time to display the graphics and photos.

# # Comparing Late_Price_Declare and Long_Loading_Time

# In[37]:


plt.figure(figsize=(13,6))
plt.title('Checking which webapp declares late price and takes longer time in loading the page',fontsize=15)
sns.countplot(x='Late_Price_Declare',hue='Long_Loading_Time',palette="cool_r",data=df);
plt.xticks(rotation=90)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),title="Long_Loading_Time")
plt.show()


# # Observation
# 
# * When there is promotion and sales, Myntra takes time ti load the page and it has late declaration of price in these days.
# * Myntra declare the late price in order to clear the sales and they fix the price by comparing with other websites and they end up sales by providing benefits to the customers. In this time most of the customers tries to shop in this website so it takes long loading time.

# # Comparing Late_Price_Declare and Long_Loading_Time

# In[38]:


plt.figure(figsize=(13,6))
plt.title('Checking which webapp gives limited mode of payment and late delivery',fontsize=15)
sns.countplot(x='Limited_Payment_Mode',hue='Late_Delivery',palette="autumn",data=df);
plt.xticks(rotation=90)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),title="Late_Delivery")
plt.show()


# # Observation
# 
# * Snapdeal has limited mode of payment on most of the products followed by Amazon. And paytm takes more time to deliver the product. So this website may not satisfy the customers due to late delivery.

# # Taking care of categorical columns using label encoding method

# In[40]:


from sklearn.preprocessing import LabelEncoder

LE=LabelEncoder()
for i in df.columns:
    if df[i].dtypes=="object":
        df[i]=LE.fit_transform(df[i])


# # Checking the dataframe after encoding

# In[41]:


df.head()


# # Checking statistical summary of the dataset

# In[43]:


df.describe()


# # Observations
# 
# Before we got only one column's statistical summary, after label encoding we can able to notice all the columns statistical summary.
# 
# * Here the count of all the columns are same which means there are no missing values present in the dataset.
# * Some of the columns have their mean value greater than the median (50%), so we can say they are skewed to right.
# * In some of the columns, the median is greater than the mean, so the data is skewed to left.
# * We can also notice the min value, standard deviation and 25% percentile.
# * In summarising the data, we cna notice huge difference between max and 75% percentile in some of the columns which means there are huge outliers present in those columns. Since all the columns in the dataset are categorical, no need to remove outliers and skewness.

# # Checking skewness in the data

# In[45]:


# Checking the skewness
df.skew()


# # Observations:
# The outliers present in many of the columns but the dataset contains all the categorical data so no need to remove the outliers.
# Skewness is also present in many of the columns but all the columns are categorical so no need to remove skewness also.

# # Identifying the outliers

# In[44]:


# Let's check the outliers by ploting box plot

plt.figure(figsize=(25,35),facecolor='white')
plotnumber=1
for column in df:
    if plotnumber<=71:
        ax=plt.subplot(15,5,plotnumber)
        sns.boxplot(df[column],palette="Set2_r")
        plt.xlabel(column,fontsize=20)
    plotnumber+=1
plt.tight_layout()


# # Checking Correlation between the independent variables

# In[46]:


# Checking the correlation between features and the target
cor = df.corr()
cor


# # Visualizing the Correlation Matrix by Plotting Heat Map.

# In[47]:


plt.figure(figsize=(40,35))
sns.heatmap(df.corr(),linewidths=.1,vmin=-1, vmax=1, fmt='.1g',linecolor="black", annot = True, annot_kws={'size':10},cmap="cubehelix_r")
plt.yticks(rotation=0);


# # Observation
# 
# This heatmap shows the correlation matrix by visualizing the data. we can observe the relation between one feature to other.
# 
# This heat mapcontains both positive and negative correlation.
# 
# * Dark shades are highly correlated.
# * Light shades are less correlated.
# * By looking at the heat map we can observe most of the columns have strong correlation with each other, which leads to multicollinearity issue and it will impact on the model accuracy, so we can check the VIF values to solve this issue by dropping the columns having VIF values more than 10. But here we are not building machine learning models so I am keeping this issue as it is.
# 

# In[ ]:


# Pie charts for some of the features 

for i in rating_columns:
    plt.subplots()
    plt.pie(x=data[i].value_counts(),labels=data[i].value_counts().index,data=data,shadow=True, 
            startangle=60,autopct='%1.1f%%',colors=['orange', 'cyan','beige','limegreen', 'magenta', 'crimson'],
            wedgeprops = {'linewidth': 4})
    plt.setp(plt.title(i,fontsize=15,color='darkred'),color='blue',style='italic')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()
    print("----------"*10)
    print('\n')
    #break

    
#Let's also use the coded data for this part

coded_data = pd.read_excel('customer_retention_dataset.xlsx','codedsheet')
df = pd.read_excel('customer_retention_dataset.xlsx','datasheet')

rating_cols=df.iloc[:,17:47].columns
rating_cols=rating_cols.to_list()

ratings = {1: 'Strongly disagree', 2: 'Disagree', 3: 'Neither agree nor disagree', 4: 'Agree', 5: 'Strongly agree'} 


# Rename the values of the dataframe 
for col in rating_cols:
    coded_data.replace({col: ratings},inplace=True)
    

import plot_likert

plot_likert.plot_likert(coded_data[rating_cols], plot_likert.scales.agree, plot_percentage=True,figsize=(10,25))
plt.show()


# # Outcomes from the data analysis:
# * In this project we have investigated ecommerce quality in online businesses and develop new knowledge to understand the most important dimensions of E-retail factor for customer activation and retention.
# * This project aimed to enhance prior understanding of how ecommerce websites affected customer satisfaction, customer trust, and customer behaviour, i.e., repurchase intention, customer loyalty, and site revisit.
# * The dimensions like information about the products, convenient payment mode, Trust, Fulfilment, website design change, security/privacy and many others had a positive impact on the ecommerce websites for customers. Also, some of the dimensions like ease of navigation, loading and speed, late delivery etc did not have impact on the ecommerce websites.
# * Thus, a company needs to pay attention to these dimensions more specifically and seek breakthroughs that can improve its performance and e-service quality.
# * Customer satisfaction and customer trust appeared as the outcomes of overall e-retail factor. The results of the analysis showed that e-retail factor had a positive impact on customer satisfaction. The majority of research done about e-retail factor states that customer satisfaction is the main determinant impacting on e-retail factor. It supports the idea that there is a significant relationship between e-retail factor and customer satisfaction. E-retail factor also had a positive impact on customer trust. The better the e-retail factor of a company, the higher the customer trust. Providing good service quality enhances customer satisfaction and customer trust.
# * From the above analysis we found that the mains reasons or factors which attract consumers to do shopping online and then main reasons or obstacles which discourage consumers from shopping online. Therefore, from the analysis, it is found that most of the respondents use internet daily but most of the respondents do not use internet daily to buy products. Nearly half of the total respondents' opinions were that they would only use the internet to buy products when the need arises to do so.

# # Summary
# Comparing the Customer's Perceptions and the Company's performance we can conclude that the Companies likely to have
# 
# * High Customer Satisfaction and Retenton:
#       * Amazon.com
#       * Flipkart.com
# 
# 
# * High Risk of Customer Churn:
#       * Myntra.com
#       * Snapdeal.com

# # Conclusions:
# Amazon -
#  The most recommended websites with attractive web-page layout, easy to use, relevant descriptive information, product offers, reliability of website, quickness to complete purchase, trust worthiness.
# 
# Things to be improved :
#  Takes longer time to login, Late declaration or price during sales and promotion, frequent disruption when moving from one page to another, Limited mode of payment on most of products.
# 
# Flipkart –
#  This is the 2nd most recommended website with fast loading page, security of financial information, trust worthiness, several payments modes, website is as efficient as before.
# 
# Things to be improved :
#  Takes longer time in displaying graphics, late declaration of price during sales and promotion.
# 
# Paytm –
#  Reliability of website, speedy delivery of products, quickness in purchase.
# 
# Things to be improved :
#  Longer page loading time, Longer delivery period , late declaration of price during sales and promotion.
# 
# Myntra –
#  Myntra stands on 3rd most recommended websites with easy to use, wild variety of product offers, several payment methods, attractive visual appealing web-page layout.
# 
# Things to be improved :
#  Relevant information about product, website loading speed, speedy delivery of products, websites is not much efficient as before.
# 
# Snapdeal –
#  Least recommended website having less page loading time.
# 
# Things to be improved :
#  Limited mode of payments, frequent disruption while moving from one page to another, Longer delivery period, customer’s privacy information, reliability of website, offers on product, and must be an attractive web-page layout.
# 
#  

# In[ ]:




