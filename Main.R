############################################# 
# MAIN SCRIPT TO DESIGN ANALYTICAL WORKLOW
# Ivana Aleksovska
#############################################

# This part contains the packages and the libraries that will be used in the following

# Packages for machine learning and data analysis
# install.packages(caret)
# install.packages(factoextra)

# Packages for Rmarkdown generator
# install.packages('knitr')
# install.packages("/Users/moia/Downloads/knitr_1.47.tar.gz", repos = NULL, type="source")

library(tidyverse)
library(caret)
library(party)
library(factoextra)
library("corrplot")
library(gridExtra)
library(grid)
library(ggplot2)
library(lattice)

# ------------------------------------------------------------------------------------------------
# 1. DEFINE THE QUESTION
# Develop tailored and climate smart fertilizer advice for maize growners in Nigeria
# Consider the yield response to N,P and K in the data and the spatial variation in 
# soil and weather in the region to predict site specific N, P and K rates.
 
# ------------------------------------------------------------------------------------------------
# 2. DOWNLOAD AND PREPARE THE DATA FOR ML 

# I) First get the data
# Follow the steps explained here to get the data
# https://carob-data.org/compile.html

# 1) online
#install.packages("remotes")
#remotes::install_github("reagro/carobiner", force = TRUE)
#remotes::install_github("reagro/carobiner")
#ff <- carobiner::make_carob("/Users/moia/Desktop/github/carob")
# carobiner::update_terms()
# 2) or zip format
# install.packages("/Users/moia/Downloads/geodata_0.5-9.tar.gz", repos = NULL, type="source")
# install.packages("/Users/moia/Desktop/github/carob/carobiner", repos = NULL, type="source")

# LOAD THE DATA IN THE RSTUDIO WORKING ENVIRONMENT
carob_fertilizer_terms <- readr::read_csv("Desktop/github/carob/carob/data/compiled/carob_fertilizer_terms.csv",show_col_types = FALSE)
carob_fertilizer_metadata <- readr::read_csv("Desktop/github/carob/carob/data/compiled/carob_fertilizer_metadata.csv",show_col_types = FALSE)
carob_fertilizer <- readr::read_csv("Desktop/github/carob/carob/data/compiled/carob_fertilizer.csv",show_col_types = FALSE)

# SELECT THE DATA OF MAIZE THAT CORRESPONDS TO THE REGION OF NIGERIA
colnames(carob_fertilizer) # see all columns available in the data frame
carob_fertilizer<- carob_fertilizer[ which(carob_fertilizer$country=="Nigeria" & carob_fertilizer$crop =="maize") , ]

# ------------------------------------------------------------------------------------------------
# 3. DATA ANALYSIS (descriptive statistic and visualisation)
# a) Perform a comprehensive exploratory data analysis to summarize the main characteristics of the data)
# b) Use appropriate visualizations to illustrate your findings. 

# In this context you should consider the yield response to N, P and K in the data 
# and the spatial variation in soil and weather in the region to predict site specific N, P, and K rates.

# In short, fertilizers are labeled N, P or K to indicate their nutrient content in terms 
# of nitrogen (N), phosphorus (P), and potassium (K). All three are important for plant growth.
# N_fertilizer: numeric (kg/ha) N applied in inorganic fertilizer
# P_fertilizer: numeric (kg/ha) P applied in inorganic fertilizer (as P, not P2O5)
# K_fertilizer: numeric (kg/ha) K applied in inorganic fertilizer (as K, not K2O)
# yield (kg/ha)

# ------
# Identify the important predictors X that influence the yield
# predictors=c('latitude','longitude','N_fertilizer','P_fertilizer','K_fertilizer','yield')
predictors=c('N_fertilizer','P_fertilizer','K_fertilizer','yield')

# from the carob_fertilizer select only those columns that will be used into ML algo
carob_fertilizer_ML=carob_fertilizer[predictors]

# Before all clean the NA (to avoid missing values in the statistics) 
# and the remove the duplicate values in the data frame 
# (this comes usually from errors in savings, and it introduce bias since double)
carob_fertilizer_ML <-na.omit(carob_fertilizer_ML)
carob_fertilizer_ML <- carob_fertilizer_ML[!duplicated(carob_fertilizer_ML),]
carob_fertilizer_ML=as.data.frame(carob_fertilizer_ML) # convert to data frame

# ------
# a) and b)

# Visual inspection / descriptive statistics
# Let's see the main statistics in the data
summary(carob_fertilizer_ML)

# Histograms of all variables in our data set, the target, labeled “yield” and the predictors, K, N and P fertilizers.
# Here you can see the univariate distributions, one variable at a time. We can visually check the type of distribution, 
# whether it's normal or not. In addition, we can check whether they need some form of normalization or discretization. 
# All these variables are continuous variables.

# See the histograms for every variable (column)
carob_fertilizer_ML %>% gather() %>%
  ggplot(aes(x=value)) + 
  geom_histogram(fill="steelblue", alpha=.7) +
  theme_minimal() +
  facet_wrap(~key, scales="free")

# Let’s move on to bivariate statistics. We are plotting a correlation matrix, in order to 
# a) check if we have predictors that are highly correlated (which is problematic for some algorithms), 
# b) get a first feeling about which predictors are correlated with the target.
# Strong correlation ~1.

corr_matrix <- cor(carob_fertilizer_ML %>% keep(is.numeric))
corr_matrix %>% as.data.frame %>% mutate(var2=rownames(.)) %>%
    pivot_longer(!var2, values_to = "value") %>%
    ggplot(aes(x=name,y=var2,fill=abs(value),label=round(value,2))) +
    geom_tile() + geom_label() + xlab("") + ylab("") +
    ggtitle("Correlation matrix of our predictors") +
    labs(fill="Correlation\n(absolute):")
  
pairs(carob_fertilizer_ML)

# Principal Component Analysis (PCA) is a statistical method designed to reduce dimensionality. 
# This is useful if we want to make a model (decision support tool) based on multiple predictors.
# PCA is used to extract meaningful information from a table of multivariate data, and to express 
# this information as a set of a few new variables called principal components. 
# The aim of PCA is to identify directions that can be visualized graphically with minimal loss of 
# information. The first principal component can be equivalently defined as a direction that maximizes 
# the variance of the projected data. Principal components are often analyzed via the covariance matrix 
# decomposition of the data or the singular value decomposition (SVD) of the data matrix.
res.pca <- prcomp(carob_fertilizer_ML[,-4], scale = TRUE) # exclude the yield column, PCA is unsupervised machine learning
print(res.pca)
summary(res.pca)
eig.val<-get_eigenvalue(res.pca)
eig.val
# On the basis of importance of components, is it visible that first two PC has 
# the highest vales for proportion of variance. This statement is also proved by eigenvalues measure. 
# They are large for the first PC and small for the subsequent PCs, which means that the first PC 
# corresponds to the directions with the maximum amount of variation in the data set. 
# As far as scatter plot is concerned, first eigenvalue explain more than 75% of the variation, 
# second ~20%. Therefore, more than 95% of the variation is explained by the first two eigenvalues 
# together, which is a proper indicator for further analysis.
fviz_eig(res.pca, col.var="blue")

# PCA results can be assesed with regard to variables. 
# Firstly, I will conduct extraction of results for variables. 
# For that purpose get_pca_var() is used to provide a list of matrices containing all the results 
# for the active variables (coordinates, correlation between variables and axes, squared cosine, 
# and contributions).

var <- get_pca_var(res.pca)
var

corrplot(var$cos2, is.corr=FALSE)
fviz_cos2(res.pca, choice = "var", axes = 1:2)

# Additionally, the quality of representation of variables can be draw on the factor map, 
# where cos2 values differ by gradient colors. Variables with low cos2 values will be colored 
# “darkorchid4”, medium cos2 values - “gold”, high co2 values - “darkorange”. 
# Positively correlated variables are grouped together, whereas negatively correlated variables 
# are positioned on opposite sides of the plot origin. The distance between variables and the 
# origin measures the quality of the variables on the factor map. Variables that are away from the 
# origin are well represented on the factor map.

fviz_pca_var(res.pca,
             col.var = "cos2", # Color by the quality of representation
             gradient.cols = c("darkorchid4", "gold", "darkorange"),
             repel = TRUE
)

# P_fertilizer, K_fertilizer and N_fertilizer have very high cos2, which implies a good 
# representation of the variable on the principal component. In this case variables are positioned 
# close to the circumference of the correlation circle. N_fertilizer has slightly lowest cos2, 
# which indicates that the variable is less good represented by the PCs.
# None of them is close to the center of the circle, so all are represented with the first components.

# Draw a bar plot of variable contributions for the most significant dimensions, therefore PC1 and PC2.
# Contributions of variables to PC1
a<-fviz_contrib(res.pca, choice = "var", axes = 1)
# Contributions of variables to PC2
b<-fviz_contrib(res.pca, choice = "var", axes = 2)
grid.arrange(a,b, ncol=2, top='Contribution of the variables to the first two PCs')

# Based on the pca analysis, we can reduce the dimension and build predictive statistics using 
# machine learning models on the pca components (keeping the first most important ones). 
# However, in this case, we've only taken 3 predictors, so we'll keep them for this exercise. 
# In any case, I'm happy to do more advanced machine learning models, using the full list of 
# predictors given in the data. In this case, PCA will be explored even further.

# ------------------------------------------------------------------------------------------------
# 4. MACHINE LEARNING (predictive statistics)

# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
