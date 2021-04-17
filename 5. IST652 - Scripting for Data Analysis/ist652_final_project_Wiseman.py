
"""
Video Games Sales Data Analysis
Created for IST 652 M800 final project
By Kobi Wiseman
July 2020
"""

### Importing the data ###

# importing the csv file using pandas
import pandas as pd
import numpy as np
VGdata = pd.read_csv("C:/Users/User/Desktop/IST652/Final Project/vgsales.csv")
url = "https://www.kaggle.com/gregorut/videogamesales"
# unfortunately I was unable to read the csv directly from url. file should be
# downloaded and read to Python locally.




### Data Cleaning: missing values, duplicates and outliers ###

# Finding total number of missing values:
A = list(VGdata.isnull().sum(axis=0))
sum(A)
print("This Video Games sales DataFrame has a total number of", sum(A), "missing values")


# total number of rows with missing values:

B = VGdata[VGdata.isnull().any(axis=1)]
len(B)
print("There are", len(B), "records which include missing values")

A2 = [i for i in A if i > 0]
print("There are", len(list(A2)), "columns which contain missing values")


# Since 'Year' is numeric but no continouous, 
# missing values will be replaced with the most frequent value:
VGdata['Year'].value_counts() # 2009 is the most common year

# replacing missing with the value of 2009 using fill.na function:
VGdata['Year'] = VGdata['Year'].fillna(2009)

# Since 'Publisher' is categorical variable and assigning missing cases
# to a specific publisher does not make much businees sense - I will create
# a new category in this varialbe called 'unknown'.
# replacing missing with the value of 'unknown' using fill.na function:
VGdata['Publisher'] = VGdata['Publisher'].fillna('Unknown')



# total number of duplicates:
import pandas as pd
print("There are", len(VGdata[VGdata.duplicated()]), "duplicated rows")
# Since there are no duplicate rows, no records deleting is needed.



# outliers will be detected and replaced with median, using the Z-score.
# since 'Global_Sales' is computed out of the regional sales and I don't have
# interest in the absolute values of these regional sales attributes - 
# they will be replaced with percentages (out of global sales column) variables.

# transforming regional sales to a percentage info:
VGdata["NA_Sales"] = VGdata["NA_Sales"] / VGdata["Global_Sales"]
VGdata["EU_Sales"] = VGdata["EU_Sales"] / VGdata["Global_Sales"]
VGdata["JP_Sales"] = VGdata["JP_Sales"] / VGdata["Global_Sales"]
VGdata["Other_Sales"] = VGdata["Other_Sales"] / VGdata["Global_Sales"]

# forming z-scored variables:
cols = list(VGdata.columns)
cols.remove('Rank')
cols.remove('Name')
cols.remove('Platform')
cols.remove('Genre')
cols.remove('Publisher')
cols.remove('NA_Sales')
cols.remove('EU_Sales')
cols.remove('JP_Sales')
cols.remove('Other_Sales')
VGdata[cols]

for col in cols:
    col_zscore = col + '_zscore'
    VGdata[col_zscore] = ((VGdata[col] - VGdata[col].mean())/VGdata[col].std(ddof=0))
VGdata

# viewing the z-score created columns, we can see that they both reach high values 
# of z-scores, meaning there is high variance for each one of them.
# However, the absolute numbers of both year and global sales makes business sense:
# games in this data are indeed range from very early era of video games untill nowdays;
# video games can be best-sellers with sales of millions of dollars worldwide and 
# could be a failure with extremely low numbers of dollars revenue.
# Therefore, I decide not to impute extreme values but to keep them as is.
# For the linear regression, I will use the global_sales attribute as is for the
# dependent variable; for the cluster analysis I will use the normalized attribute. 



### Exploratory Data Analysis (EDA) and visualization ###

# forming a grid for a describe display
# percentile list 
perc =[.20, .40, .60, .80] 
  
# list of dtypes to include 
include =['object', 'float', 'int'] 
  
# calling describe method 
desc = VGdata[["Year","NA_Sales","EU_Sales","JP_Sales","Other_Sales","Global_Sales"]].describe(percentiles = perc, include = include) 
  
# display 
print(round(desc,2))

# video games range from 1980 to 2020. Plenty of changes in computers
# and game consoles systems technology have been occurred along years,
# so in the popular culture. Given that, I expect this variable to be a major
# key in predicting the global sales of a given video game.
# having most of its data being on the low portions among overall revenues,
# "other sales" has usually low influence compared to the defined other three
# regions. Japan and North America both reach high portion of the sales for 
# certain franchises, which implies each one the areas enjoy a separate market
# share. I am curious to see if that will be seen at the cluster analysis as well.


# histograms for categorical attributes:

import matplotlib.pyplot as plt
import seaborn as sns


platPlatform = pd.crosstab(VGdata.Platform,VGdata.Genre)
platPlatformTotal = platPlatform.sum(axis=1).sort_values(ascending = False)
plt.figure(figsize=(8,6))
sns.barplot(y = platPlatformTotal.index, x = platPlatformTotal.values, orient='h')
plt.ylabel = "Platform"
plt.xlabel = "The amount of games"
plt.show()
# Nintendo DS and PlayStation 2 are by far the most common consoles in the past
# 30 years. Far from those, but with noticeable presence, we observe PlayStation 3,
# Nintendo Wii, Xbox 360, PSP and PS with roughly 1200-1300 games each. PC, which
# at the far past used to be only source for gaming, is behind all the above,
# with less than 1000 games within the data.

platGenre = pd.crosstab(VGdata.Genre,VGdata.Platform)
platGenreTotal = platGenre.sum(axis=1).sort_values(ascending = False)
plt.figure(figsize=(8,6))
sns.barplot(y = platGenreTotal.index, x = platGenreTotal.values, orient='h')
plt.ylabel = "Genre"
plt.xlabel = "The amount of games"
plt.show()
# Action and sports games are the most common genres among the best-sellers
# during the last 40 years. Strategy and puzzles are the least frequent, 
# with only a little more than 500 games each.

'''
platPublisher = pd.crosstab(VGdata.Publisher,VGdata.Publisher)
platPublisher['Total'] = platPublisher.sum()
popPlatTotal = platPublisher[['Total']]
popPlatform = popPlatTotal['Total'].sort_values(ascending = False)
plt.figure(figsize=(14,14))
sns.barplot(y = popPlatform.index, x = popPlatform.values, orient='h')
plt.ylabel = "Publisher"
plt.xlabel = "The amount of games"
plt.show()
'''
# while working on this specific attribute's display I noticed there are many 
# game publisher with low frequency. Therefore, I decided to combine all those
# that have less than 50 games in one category named 'Other':

series = pd.Series(pd.value_counts(VGdata.Publisher))
toother = series.where(series<50)
to_other = toother[toother > 0]
VGdata["Publisher"] = np.where(VGdata["Publisher"].isin(to_other.index), "other", VGdata["Publisher"])

# running the plot after the 'other' modification:
platPublisher = pd.crosstab(VGdata.Publisher,VGdata.Publisher)
platPublisher['Total'] = platPublisher.sum()
popPlatTotal = platPublisher[['Total']]
popPlatform = popPlatTotal['Total'].sort_values(ascending = False)
plt.figure(figsize=(14,14))
sns.barplot(y = popPlatform.index, x = popPlatform.values, orient='h')
plt.ylabel = "Publisher"
plt.xlabel = "The amount of games"
plt.show()
# The 'other' category which covers all the video games studios with less than
# 50 games each, reaches over 3,000 games overall. Electronic Arts is the leading
# publisher with roughly 1,500 games, followed by Activision, Namco and Ubisoft 
# with about 1,000 games each.

### Preparation for Linear Regression Model ###
# creating a dataframe with only the relevant attributes for the linear model:
VGLin = VGdata[['Rank','Platform', 'Year', 'Genre', 'Publisher', 'NA_Sales','EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']]
VGLinFin = pd.get_dummies(VGLin, columns=['Platform', 'Genre', 'Publisher'],drop_first=True)

# moving the Global_Sales column back to the end:
cols_at_end = ['Global_Sales']
VGLinFin = VGLinFin[[c for c in VGLinFin if c not in cols_at_end] 
        + [c for c in cols_at_end if c in VGLinFin]] 
   

# checking multicollinearity

VGCorr = VGLinFin.corr()
# JP_Sales is highly correlated with NA_Sales: -0.71 . 
# JP_Sales is also highly correlated with EU_Sales: -0.49 .
# Since NA_Sales and EU_Sales are not highly correlated (-0.21) I will drop only
# the JP_Sales attribute to avoid multicollinearity.

VGLinFin = VGLinFin.drop(['JP_Sales'], axis=1)


# defining the columns for dependent and explanatory variables
X = VGLinFin.iloc[:,0:92]
Y = VGLinFin.iloc[:,92]


# setting the train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# Running the linear regression
import statsmodels.api as sm
import numpy as np
mod = sm.OLS(Y,X)
fii = mod.fit()
# adding the p_values for each coefficient:
fii.params
fii.pvalues
coeff_df = pd.DataFrame(fii.params, X.columns, columns=['Coefficient']) 
coeff_df['p-value'] = fii.pvalues
print(coeff_df.sort_values(by=['p-value'], ascending=False))
# getting the r-squared:
r2_score = round(fii.rsquared_adj,ndigits=2)
print("The adjusted r-squared value for this model is ",r2_score)
print("The MSE for this model is ",fii.mse_model)
fii.summary()



# fine-tuning:
# I will try running the same model while keeping only the significant variables.
coeff_df['Attribute'] = coeff_df.index
coeff_df.reset_index(drop=True, inplace=True)
sigdf = coeff_df[(coeff_df['p-value'] < 0.05)] # keep only rows with p-value < 0.05
siglist = sigdf[['Attribute']]
Global_Sales = VGLinFin[['Global_Sales']]

VGLin2 = pd.DataFrame(VGLinFin[siglist['Attribute']])
VGLinFin2 = pd.concat([VGLin2, Global_Sales], axis=1, sort=False)



# defining the columns for dependent and explanatory variables
X2 = VGLinFin2.iloc[:,0:22]
Y2 = VGLinFin2.iloc[:,22]


# setting the train/test split
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, Y2, test_size=0.3, random_state=0)

# Running the linear regression
import statsmodels.api as sm
mod2 = sm.OLS(Y2,X2)
fii2 = mod2.fit()
# adding the p_values for each coefficient:
fii2.params
fii2.pvalues
coeff_df2 = pd.DataFrame(fii2.params, X2.columns, columns=['Coefficient']) 
coeff_df2['p-value'] = fii2.pvalues
print(coeff_df.sort_values(by=['p-value'], ascending=False))
# getting the r-squared:
r2_score2 = round(fii2.rsquared_adj,ndigits=2)
print("The adjusted r-squared value for this model is ",r2_score2)
print("The MSE for this model is ",fii2.mse_model)
print(fii2.summary())

### K-means Cluster Analysis ###

# forming the initial dataframe for the k-means cluster analysis:
Kdata = VGdata.drop(['Name','Global_Sales','Global_Sales_zscore','Year_zscore'], axis=1) # binning rank
# and global sales by its order has the same meaning, therefore only 'Rank' will remain.

# discretizing the areal percentages variables:
# attributes are binned into 10 equal intervals.

Kdata['NA_Sales'] = pd.cut(Kdata['NA_Sales'],bins=10, labels=[1,2,3,4,5,6,7,8,9,10])
Kdata['EU_Sales'] = pd.cut(Kdata['EU_Sales'],bins=10, labels=[1,2,3,4,5,6,7,8,9,10])
Kdata['JP_Sales'] = pd.cut(Kdata['JP_Sales'],bins=10, labels=[1,2,3,4,5,6,7,8,9,10])
Kdata['Other_Sales'] = pd.cut(Kdata['Other_Sales'],bins=10, labels=[1,2,3,4,5,6,7,8,9,10])
Kdata['Year'] = pd.cut(Kdata['Year'],bins=10, labels=[1,2,3,4,5,6,7,8,9,10])
Kdata['Rank'] = pd.cut(Kdata['Rank'],bins=10, labels=[1,2,3,4,5,6,7,8,9,10])


# generating more 'business tailored' dummy variables using np.where function:
Kdata['Nintendo'] = np.where(Kdata['Platform'].isin(['Wii','WiiU','NES','SNES','GB','GBA','DS','3DS','GC','N64']), 1, 0)
Kdata['Playstation'] = np.where(Kdata['Platform'].isin(['PS','PS2','PS3','PS4','PSP','PSV']), 1, 0)
Kdata['Xbox'] = np.where(Kdata['Platform'].isin(['XB','X360','XOne']), 1, 0)
Kdata['PC'] = np.where(Kdata['Platform'].isin(['PC']), 1, 0)
Kdata['Adrenaline'] = np.where(Kdata['Genre'].isin(['Action','Adventure','Fighting','Platform','Racing','Shooter','Sports']), 1, 0)
Kdata['Logic'] = np.where(Kdata['Genre'].isin(['Puzzle','Role-Playing','Simulation','Strategy']), 1, 0)

# after creating new variables, dropping the original ones:
KdataFin = Kdata.drop(['Platform','Genre','Publisher'], axis=1)


# determining the number of clusters, using PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=5) 
pca.fit(KdataFin) 
pca_data = pd.DataFrame(pca.transform(KdataFin)) 
print(pca.explained_variance_) # 4 clusters is the ideal number of clusters
# (5th cluster has eigenvalue smaller than 1).

# running k-means cluster analysis with 4 clusters
from sklearn.cluster import KMeans
myKmeans = KMeans(n_clusters=4).fit(KdataFin)
print(myKmeans.labels_)
clusternum = myKmeans.labels_


clusternum = clusternum+1 # adding 1 so we'll have a normal number of clusters:
# 1 thru 4.
clusternum = pd.DataFrame(clusternum, columns=['cluster'])
# attaching each game's cluster number back to the original dataset:
KdataClust = pd.concat([KdataFin, clusternum], axis=1, sort=False)


# view absolute frequencies of cluster vs. each variable:
pd.crosstab(KdataClust['cluster'], KdataClust['Rank'], rownames=['cluster'], colnames=['Rank'])               
pd.crosstab(KdataClust['cluster'], KdataClust['Year'], rownames=['cluster'], colnames=['Year'])               
pd.crosstab(KdataClust['cluster'], KdataClust['NA_Sales'], rownames=['cluster'], colnames=['NA_Sales'])               
pd.crosstab(KdataClust['cluster'], KdataClust['EU_Sales'], rownames=['cluster'], colnames=['EU_Sales'])               
pd.crosstab(KdataClust['cluster'], KdataClust['JP_Sales'], rownames=['cluster'], colnames=['JP_Sales'])               
pd.crosstab(KdataClust['cluster'], KdataClust['Other_Sales'], rownames=['cluster'], colnames=['Other_Sales'])                           
pd.crosstab(KdataClust['cluster'], KdataClust['Nintendo'], rownames=['cluster'], colnames=['Nintendo'])               
pd.crosstab(KdataClust['cluster'], KdataClust['Playstation'], rownames=['cluster'], colnames=['Playstation'])               
pd.crosstab(KdataClust['cluster'], KdataClust['Xbox'], rownames=['cluster'], colnames=['Xbox'])               
pd.crosstab(KdataClust['cluster'], KdataClust['PC'], rownames=['cluster'], colnames=['PC'])               
pd.crosstab(KdataClust['cluster'], KdataClust['Adrenaline'], rownames=['cluster'], colnames=['Adrenaline'])               
pd.crosstab(KdataClust['cluster'], KdataClust['Logic'], rownames=['cluster'], colnames=['Logic'])               

# view by percentages of each cluster's total:
ctRank = pd.crosstab(KdataClust.cluster, KdataClust.Rank).apply(lambda r: r/r.sum(), axis=1)
ctYear = pd.crosstab(KdataClust.cluster, KdataClust.Year).apply(lambda r: r/r.sum(), axis=1)
ctNA_Sales = pd.crosstab(KdataClust.cluster, KdataClust.NA_Sales).apply(lambda r: r/r.sum(), axis=1)
ctEU_Sales = pd.crosstab(KdataClust.cluster, KdataClust.EU_Sales).apply(lambda r: r/r.sum(), axis=1)
ctJP_Sales = pd.crosstab(KdataClust.cluster, KdataClust.JP_Sales).apply(lambda r: r/r.sum(), axis=1)
ctOther_Sales = pd.crosstab(KdataClust.cluster, KdataClust.Other_Sales).apply(lambda r: r/r.sum(), axis=1)
ctNintendo = pd.crosstab(KdataClust.cluster, KdataClust.Nintendo).apply(lambda r: r/r.sum(), axis=1)
ctPlaystation = pd.crosstab(KdataClust.cluster, KdataClust.Playstation).apply(lambda r: r/r.sum(), axis=1)
ctXbox = pd.crosstab(KdataClust.cluster, KdataClust.Xbox).apply(lambda r: r/r.sum(), axis=1)
ctPC = pd.crosstab(KdataClust.cluster, KdataClust.PC).apply(lambda r: r/r.sum(), axis=1)
ctAdrenaline = pd.crosstab(KdataClust.cluster, KdataClust.Adrenaline).apply(lambda r: r/r.sum(), axis=1)
ctLogic = pd.crosstab(KdataClust.cluster, KdataClust.Logic).apply(lambda r: r/r.sum(), axis=1)

# visualization of the percentage output:
ctRank.plot.bar(stacked=True) 
plt.legend(title='Rank', bbox_to_anchor=(1.05,1))
plt.show()

ctYear.plot.bar(stacked=True) 
plt.legend(title='Year', bbox_to_anchor=(1.05,1))
plt.show()

ctNA_Sales.plot.bar(stacked=True) 
plt.legend(title='NA_Sales', bbox_to_anchor=(1.05,1))
plt.show()

ctEU_Sales.plot.bar(stacked=True) 
plt.legend(title='EU_Sales', bbox_to_anchor=(1.05,1))
plt.show()

ctJP_Sales.plot.bar(stacked=True) 
plt.legend(title='JP_Sales', bbox_to_anchor=(1.05,1))
plt.show()

ctOther_Sales.plot.bar(stacked=True) 
plt.legend(title='Other_Sales', bbox_to_anchor=(1.05,1))
plt.show()

ctNintendo.plot.bar(stacked=True) 
plt.legend(title='Nintendo', bbox_to_anchor=(1.05,1))
plt.show()

ctPlaystation.plot.bar(stacked=True) 
plt.legend(title='Playstation', bbox_to_anchor=(1.05,1))
plt.show()

ctXbox.plot.bar(stacked=True) 
plt.legend(title='Xbox', bbox_to_anchor=(1.05,1))
plt.show()

ctPC.plot.bar(stacked=True) 
plt.legend(title='PC', bbox_to_anchor=(1.05,1))
plt.show()

ctAdrenaline.plot.bar(stacked=True) 
plt.legend(title='Adrenaline', bbox_to_anchor=(1.05,1))
plt.show()

ctLogic.plot.bar(stacked=True) 
plt.legend(title='Logic', bbox_to_anchor=(1.05,1))
plt.show()

# printing outputs for final paper:
print(round(ctRank,2))
print(round(ctYear,2))
print(round(ctNA_Sales,2))
print(round(ctEU_Sales,2))
print(round(ctJP_Sales,2))
print(round(ctOther_Sales,2))
print(round(ctNintendo,2))
print(round(ctPlaystation,2))
print(round(ctXbox,2))
print(round(ctPC,2))
print(round(ctAdrenaline,2))
print(round(ctLogic,2))