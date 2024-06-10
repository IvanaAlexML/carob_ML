############################################# 
# MAIN SCRIPT TO DESIGN ANALYTICAL WORKLOW
# Ivana Aleksovska
#############################################

# This part contains the packages and the libraries that will be used in the following

# Packages for machine learning and data analysis
# install.packages(caret)
# install.packages(factoextra)
# install.packages(Metrics)

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
library(Metrics)
# ---------------------------------------------------------------------------------------
# 1. DEFINE THE QUESTION
# Develop tailored and climate smart fertilizer advice for maize growners in Nigeria
# Consider the yield response to N,P and K in the data and the spatial variation in 
# soil and weather in the region to predict site specific N, P and K rates.
 
# ---------------------------------------------------------------------------------------
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
carob_fertilizer_terms <- readr::read_csv("data/compiled/carob_fertilizer_terms.csv",show_col_types = FALSE)
carob_fertilizer_metadata <- readr::read_csv("data/compiled/carob_fertilizer_metadata.csv",show_col_types = FALSE)
carob_fertilizer <- readr::read_csv("data/compiled/carob_fertilizer.csv",show_col_types = FALSE)

# SELECT THE DATA OF MAIZE THAT CORRESPONDS TO THE REGION OF NIGERIA
colnames(carob_fertilizer) # see all columns available in the data frame
carob_fertilizer<- carob_fertilizer[ which(carob_fertilizer$country=="Nigeria" & carob_fertilizer$crop =="maize") , ]

# ---------------------------------------------------------------------------------------
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

# Visual inspection / descriptive statistics
# Let's see the main statistics in the data. Here we are interested into mean, min, max values, and also for some quantiles.
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
  
#Let's also visualize the pairwise correlation graph using the pairs function, to see the correlation between each pair of available data, i.e. between the target and each input, but also to study each pair of inputs.
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

# ---------------------------------------------------------------------------------------
# 4. MACHINE LEARNING (predictive statistics)

# I) Partition data into training and test sets (usually also validation set, but here for simplicity only 2). We use 70% of the data for training, and 30% for testing.
set.seed(2022)
split <- sample(1:nrow(carob_fertilizer_ML), as.integer(0.7*nrow(carob_fertilizer_ML)), F)
train <- carob_fertilizer_ML[split,]
test <- carob_fertilizer_ML[-split,]

# Here, we don't pre-process the data. These are continuous, non-categorical variables. What's more, since their units are the same kg/h, no normalization is required. 
# From the train and test data, select the predictors x (fertilizers), and the target y (yield)
x_predictors=predictors[predictors!="yield"]

x_train=train[,x_predictors]
x_test=test[,x_predictors]

y_train <- train[,"yield"]
y_test <- test[,"yield"]

# ---------------------------------------------------------------------------------------
# II) Visualize exemplary algorithm

#Visualize exemplary algorithm. This step helps you understand what is going on when you subsequently train a more complex algorithm on your data. Let's strart with a decision tree here, because this is the foundation of more complex algorithms such as random forests or neural network. From the simple decision tree on our training data we can implement the following. Starting at the top, the most important predictor that can divide the data into the two most dissimilar subsets (in terms of expected yields) is “N_fertilizer”, i.e. whether the amount applied is greater or less than 66 kg/hr. If the quantity applied is greater (value > 66), we continue on the right-hand branch of the tree, otherwise we continue on the left-hand branch. If the quantity applied is greater (value > 66), we move on to the right-hand branch of the tree, otherwise we move on to the left-hand branch. For the right branch, the next important predictor is K_fertilizer, and the threshold to be considered here is: either <=41 (which leads to node 12 with a higher yield value), or >41.5 (which leads to node 13 with lower expected yield quantities). For the left branch, i.e. “N_fertilizer” <=66, the next predictor is K_fertilizer (with significant thresholds <=20 and >20 kg/h). When the K fertilizer applied is <=20, we must also consider P fertilizer as an option, with a significant threshold of 4.367 kg/h (a lower amount leads to node 4 and a higher amount leads to nodes 6 and 7). The combinations of different fertilizers with associated thresholds that lead to the worst yield prediction are associated with node 7. In the training data, there were 36 cases. The box-plots on the bottom of the chart show the statistics of the recored yields in the training data for the combination of different fertilizers and the corresponding applied quantitiy.

# DECISION TREE
set.seed(2024)
tree1 <- party::ctree(y_train ~ ., data=cbind(x_train, y_train), 
                      controls = ctree_control(minsplit=10, mincriterion = .999))
plot(tree1)

# ---------------------------------------------------------------------------------------
# III) Machine learning models

# a) RANDOM FOREST

#First we will run random forest which is basically an ensemble of many trees as the one we built in the previous section. The trick is that each tree is grown with only a random subset of all precitors considered at each node, and in the end all trees take a vote how to classify a specific patient. Taking a subset of all precitors at each run ensures that the trees are less correlated, i.e. not all of them use the same rules as the example tree shown above. If there are a few dominant precitors (N_fertilizer), then there will be some trees in our forest grown without these dominant precitors. These trees will be better able to classify the subgroup of predictors, for whatever reasons. Our first tree in the previous section would be confused about what to predict for yields associated for those predictors. However, in the random forest, there are trees that understand these cases as well. Thus, an ensemble of learners such as a random forest most often outperforms a single learner.

# Let's set the hyperparamter via the tuneGrid argument. The “mtry” parameter specifies how many of the precitors to consider at each split. We have 3 precitors in our training dataset, so if you set mtry == 3, then it’s not a random forest any more, because all precitors are used and no random selection is applied. If you set mtry == 1, then the trees will be totally different from each other, but most ones will perform poorly because they are forced to use certain variables at the top split which are maybe not useful. The lower mtry, the more decorrelated the trees are, and the higher the value, the more precitors each tree can consider and thus the better the performance of a single tree. Somewhere between 1 and 3 is the optimal value, and there is no theoretical guidance as to which value should be taken. It depends on your data at hand, how correlated the precitors are, whether there are distinct sub-groups where the causal structure of the precitors works differently, and so on. The ideal will be to use mtry=2. Since for this exercise we have only 3 predictors, I will use mtry = c(1,2,3), which means I will make random forecast, from random trees that have all possible combinatons of predictors. Finding the best performance for different match for tuning hyperparameters is out of scope for this exercices.

# In “trainControl” we will specify “method = ‘cv'” which stands for “cross validation”. This means we will create another random split, splitting the training data into training and validation sets for the purpose of determining which algorithm works best on the training data. We set “number = 5”, the function creates a validation set of size 1/5 of x_train and takes 4/5 of the data for training. “Cross validation” therefore repeats this training process and changes the validation set to another fifth of the data. This is done 5 times in total, so that all parts of the data served as validation set once, and then the results are averaged. This routine thus lets you use all of your training data and still have train/validation splits in order to avoid overfitting.

set.seed(2024)
garbage <- capture.output(mod_rf <- caret::train(x_train, y_train, method="rf", 
                    tuneGrid = expand.grid(mtry = seq(1,ncol(x_train),by=1)),
                    trControl = trainControl(method="cv", number=5, verboseIter = T),
                    verbose = FALSE))

# The summary of our random forest model "mod_rf" is to check if the data were set. There are 3 predictors, 4487 samples. Three values were tried for the hyperparameter “mtry”. With each of the values, 5-fold cross validation was performed. If you look at the accuracy values, all values for mtry gives the same performances with our data. On average (of the five runs using cross-validation), for each mtry value, the accuracy was obtained with a train/validation split into the training data, and the performance is expressed via RMSE, R2 and MAE.
mod_rf

#Feature importance plot tells which of the variables were most often used as the important splits at the top of the trees. Unlike the single decision tree on all of the training data, where “N_fertilizer” was the most important feature, across an ensemble of different trees, it’s actually “P_fertilizer”. 

plot(varImp(mod_rf), main="Predictor importance of random forest model on training data",cex=0.5)

# b) NEURAL NETWORK

# Let’s try out a neural network. Those deep learning methods can be useful for a complex relationship between predictors and targets. 

#Here we use the pre-processing steps of centering and scaling the data because,neural networks are optimized more easily if the predictors have similar numerical ranges. Near-zero variance (“nzv”) means that we disregard predictors where almost all fertilizers have the same value. Tree-based methods such as random forests are not as sensitive to these issues.

#We have a few more tuning parameters here. “Size” refers to the number of nodes in the hidden layer. Our network has an input layer of 3 nodes (i.e. the number of predictors) and an output layer with one node (target yield) and in between, a hidden layer where interactions between the predictors and non-linear transformations can be learned. As with other hyperparameters, the optimal size of the hidden layer(s) depend on the data, so we just try the same set of values like in RF. Decay is a regularization parameter that causes the weights of our nodes to decrease a bit after each round of updating the values after backpropagation (i.e. the opposite of what the learning rate does wich is used in other implementations of neural networks). What this means is, roughly speaking, we don’t want the network to learn too ambitiously with each step of adapting its parameters to the evidence, in order to avoid overfitting. We have passed 3 different values for “size” to consider, 4 values for “decay”, and two for “bag” (true or false, specifying how to aggregate several networks’ predictions with various random number seeds, which is what the avNNet classifier does, bagging = bootstrap aggregating), so we have 3*4*2 = 24 combinations to try out.

set.seed(2024)
garbage <- capture.output(mod_nn <- caret::train(x_train, y_train, method="avNNet",
                                                 preProcess = c("center", "scale", "nzv"),
                                                 tuneGrid = expand.grid(size = seq(1,3,by=1), decay=c(1e-03, 0.01, 0.1,0),bag=c(T,F)),
                                                 trControl = trainControl(method="cv", number=5, verboseIter = T),
                                                 importance=T))
mod_nn


#As before we use predictor importance plot. Here, “N_fertilizer” was the most often used predictor.

plot(varImp(mod_nn), main="Feature importance of neural network classifier on training data",cex=0.5)


# C) LINEAR REGRESSION
lm.fit=lm(yield~N_fertilizer+P_fertilizer+K_fertilizer, data=train)
summary(lm.fit)

regressControl  <- trainControl(method="repeatedcv",
                                number = 4,
                                repeats = 5
) 

mod_lm <- train(yield~N_fertilizer+P_fertilizer+K_fertilizer,
                 data = train,
                 method  = "lm",
                 trControl = regressControl, 
                 tuneGrid  = expand.grid(intercept = FALSE))

# ---------------------------------------------------------------------------------------

# IV) Compare the performance of the three algorithms:

#Root Mean Square Error (RMSE) is the standard deviation of the residuals (prediction errors). Residuals are a measure of how far from the regression line data points are; RMSE is a measure of how to spread out these residuals are. In other words, it tells you how concentrated the data is around the line of best fit. RMSE value with zero indicates that the model has a perfect fit. The lower the RMSE, the better the model and its predictions. A higher RMSE indicates that there is a large deviation from the residual to the ground truth. From the histogram we can see that the RF outperform NN and LM.
results <- data.frame(Model = c(mod_rf$method,mod_nn$method, mod_lm$method),
                        RMSE = c(max(mod_rf$results$RMSE), max(mod_nn$results$RMSE), max(mod_lm$results$RMSE)))
results %>% ggplot(aes(x=Model, y=RMSE, label=paste(round(RMSE)))) +
  geom_col(fill="steelblue") + theme_minimal() + geom_label() +
  ggtitle("RMSE in the training data by algorithm")

#R-Squared (R² or the coefficient of determination) is a statistical measure in a regression model that determines the proportion of variance in the dependent variable that can be explained by the independent variable. The standards for a good R-squared is to be close to 1. From the histogram we can see that the RF outperform NN and LM.

results <- data.frame(Model = c(mod_rf$method,mod_nn$method, mod_lm$method),
                      Rsquared = c(max(mod_rf$results$Rsquared), max(na.omit(mod_nn$results$Rsquared)), max(mod_lm$results$Rsquared)))
results %>% ggplot(aes(x=Model, y=Rsquared, label=paste(round(Rsquared,1)))) +
  geom_col(fill="steelblue") + theme_minimal() + geom_label() +
  ggtitle("R2 in the training data by algorithm")

# MAE stands for mean absolute error, which is a measure of how close your predictions are to the actual values in a regression problem. It is calculated by taking the average of the absolute differences between the predicted and the actual values for each observation in your data set. The closer MAE is to 0, the more accurate the model is. From the histogram we can see that the RF outperform NN and LM.

results <- data.frame(Model = c(mod_rf$method,mod_nn$method, mod_lm$method),
                      MAE = c(max(mod_rf$results$MAE), max(mod_nn$results$MAE), max(mod_lm$results$MAE)))
results %>% ggplot(aes(x=Model, y=MAE, label=paste(round(MAE,1)))) +
  geom_col(fill="steelblue") + theme_minimal() + geom_label() +
  ggtitle("MAE in the training data by algorithm")

# REMARK! Even if RF shows better performance than NN and LM, I would still look for a model that gives better accuracy on the training data, i.e. R2 closer to 1 and RMSE and MAE closer to 0. This means that we should either include more predictors in the machine learning models, or think about another model that will be more appropriate for the data.

# ---------------------------------------------------------------------------------------
# V) EVALUATION ON TEST DATA

#Model evaluation against the test data. We now compare our model’s prediction against the reserved test dataset. We use the random forest to predict the test data, and we compare the predictions against the actual outcomes. 

predictions <- predict(mod_rf, newdata = x_test,na.action = na.pass)

RMSE=sqrt(mean((y_test - predictions)^2))
RMSE

MAE=mae(y_test, predictions)
MAE

rss <- sum((predictions - y_test) ^ 2)  ## residual sum of squares
tss <- sum((y_test - mean(y_test)) ^ 2)  ## total sum of squares
R2 <- 1 - rss/tss
R2
#The evaluation metrics RMSE, MAE and R2 shows similar behaviour on test data as on training data.

# or using the confusion matrix and the mesures like precision, recall and F_meas for classification machine learning models (different output that regression) (https://www.rdocumentation.org/packages/yardstick/versions/1.3.1/topics/precision)

# predicted <- factor(predict(mod_rf, x_test))
# real <- factor(y_test)
# 
# my_data1 <- data.frame(data = predicted, type = "prediction")
# my_data2 <- data.frame(data = real, type = "real")
# my_data3 <- rbind(my_data1,my_data2)
# 
# str(my_data3)
# # Check if the levels are identical
# identical(levels(my_data3[my_data3$type == "prediction",1]) , levels(my_data3[my_data3$type == "real",1]))
# confMatrix=confusionMatrix(my_data3[my_data3$type == "prediction",1], my_data3[my_data3$type == "real",1],  dnn = c("Prediction", "Reference"))
# 
# head(confMatrix$byClass)
# 
# precision(my_data3[my_data3$type == "prediction",1], my_data3[my_data3$type == "real",1])
# recall(my_data3[my_data3$type == "prediction",1], my_data3[my_data3$type == "real",1])
# F_meas(my_data3[my_data3$type == "prediction",1], my_data3[my_data3$type == "real",1])

#Confusion matrix and summary statistics of our predictions on the test set. In addition to accuracy, other metrics are often used to evaluate the goodness of a machine-learning algorithm. You can resort to using sensitivity/specificity which are also given in the output (specificity = how many of the true positive cases are detected, which is a useful indicator if the positive cases are rare, and specificity = how many true negatives are correctly classified). Which of these metrics is more important to you depends on your case, i.e. your cost function. In this case, I’d say it’s better to detect all true cases who have the disease, and we can live with a few false positives, so I’d look at sensitivity rather than specificity. In other cases, you want to avoid many false positives (e.g., spam detection, it’s much more annoying if many of your important work e-mails disappear in the spam folder), so sensitivity is maybe more important. In addition to these metrics, you also often find precision (proportion of true positive predictions relative to all “positive” predictions), and recall (proportion of true positive predictions relative to all actual positives), and F1 (harmonic mean of precision and recall). 

# ---------------------------------------------------------------------------------------
# VI) MODEL PREDICTIONS

# Once we have a good fitted model, we can make predictions for any set of predictors. Suppose a farmer is interested in predicting yields and plans to apply the following amounts of fertilizer. 
new_data=data.frame(N_fertilizer=120,P_fertilizer=35,K_fertilizer=50)
predictions <- predict(mod_rf, newdata = new_data[1,],na.action = na.pass)
predictions

# ---------------------------------------------------------------------------------------
