<img src='https://imarticus.org/wp-content/uploads/2018/07/machine-learning.png'>

<h2><b><i>Simple Linear Regression using Python</i></b></h2>

* Also known as Univariate Regression
* One Dependent Variable and One Independent Variable (Say X and y)
* Task is to find and fit the equation y = b0 + b1X 
* Should find best fit such a way that all datapoints are less deviated from line (Method of least squares)
  * Calculating the distance between each point and line will helps you in fitting the best line [y-y^]^2
* Regression comes under Supervised Learning. Here we give both x and y(I/P and O/P) to build a model
<img src ='https://seaborn.pydata.org/_images/seaborn-regplot-1.png'>
<b>Simple steps followed :</b>
<ul> <li> Understanding the math behind Simple Linear Regression and need of this concept</li>
<li> Import required Libraries</li>
<li> Get the Data set</li>
<li> If needed, perform pre-processing task</li>
<li> Identify and define Dependent and Independent variables</li>
<li> Split the data (train data, test data)</li>
<li> Fit the model for train data</li>
<li> Predict for test data using the model</li>
<li> Visualization</li> </ul>
 

<h2><b><i>Logistic Regression using Python</i></b></h2>
<img src = 'https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Preprocessing+ML/content_lr_2.png'>
<ol>
<li>To be precise linear regression is actually the underlying concept that is used in Logistic regression
<li>Understanding why Logistic regression is called as classifier </ol>

<b>Simple steps followed :</b>
<ul> <li> Understanding the math behind Logistic Regression and need of this concept. It is obvious logistic Regression is a classifier.
   Most preferrable when we perform Binary Classification</li>
<li> Import required Libraries</li>
<li> Get the Data set</li>
<li> If needed, perform pre-processing task</li>
<li> Identify and define target and Independent variables</li>
<li> Split the data (train data, test data)</li>
<li> Fit the model for train data</li>
<li> Predict for test data using the model</li>
<li> Visualization for train data and for test data as well so that it is easy to see which are classified correctly by our model
   In my code I used meshgrid to visualize the classification. Please Google for meshgrid.</li> </ul>


<h2><b><i>Decision Tree Classifier using Python</i></b></h2>
<ul><li> Most used classification Algorithm in Machine learning
<li> Decision Node(Head node) is decided by using either of Gini Index or Information gaining</li>
<li> Information Gain is built on mathematical concept Entropy. Simply Entropy is nothing but Degree of Randomness 
     Σ(-plog p)</li>
<li> <b>CART(Classification and Regression Trees)</b> is one of the decision tree algorithms and it uses GINI index based impurity            measure</li>
<li> <b>Formula</b> Gini = 1 - Σ(P suffix i) [ for i = 1 to no.of classes in the feature ] </li>
<li> Calculate Gini index for all the dependent features and choose dependent feature with lowest Gini index as Head Node</li>
<li> <b>Example (Gini index calculation for a feature be like)</b></li>
 
    sunny_yes = len( data[(data.Outlook=='Sunny') & (data.Play=='Yes')])
    sunny_no = len(data[(data.Outlook=='Sunny') & (data.Play=='No')])
    sunny = len(data[data.Outlook=='Sunny'])
    
    gini1_sunny=1-(sunny_yes/sunny)**2-(sunny_no/sunny)**2
    gini1_sunny
    
    oc_yes = len( data[(data.Outlook=='Overcast') & (data.Play=='Yes')])
    oc_no = len(data[(data.Outlook=='Overcast') & (data.Play=='No')])
    oc = len(data[data.Outlook=='Overcast'])

    gini1_overcast=1-(oc_yes/oc)**2-(oc_no/oc)**2
    gini1_overcast
    
    rain_yes = len( data[(data.Outlook=='Rain') & (data.Play=='Yes')])
    rain_no = len(data[(data.Outlook=='Rain') & (data.Play=='No')])
    rain = len(data[data.Outlook=='Rain'])

    gini1_rain=1-(rain_yes/rain)**2-(rain_no/rain)**2
    gini1_rain
    
    gini_outlook = (sunny/data.shape[0])*gini1_sunny + (oc/data.shape[0])*gini1_overcast + (rain/data.shape[0])*gini1_rain
    gini_outlook
* In the above example feature is Outlook. Sunny, overcast and rain are three classes and Gini Index is calculated for each class      and the Gini index for whole feature is calculated atlast. Like wise we calculate Gini Index for all features w.r.to target              Variable
* Finally the feature with least Gini Index is selected as Head Node
 
<b>Simple steps followed :</b>
<ul><li> Understanding the math behind Decision Tree Classifier and need of this classifier</li>
<li> Import required Libraries, from sklearn.trees import <b>DecisionTreeClassifier</b></li>
<li> Get the Data set</li>
<li> If needed, perform pre-processing task</li>
<li> Identify and define Dependent and Independent variables</li>
<li> I did scaling in my code, No need of using scaler</li> 
<li> Split the data (train data, test data)</li>
<li> Fit the model for train data</li>
<li> Hyper Tuning is very essential if you are going for numbers like accuracy score and F1</li> 
<li> Predict for test data using the model</li>
<li> Visualization : You can Visualize the tree or you can also visualize using meshgrid</li> </ul>
    
<h2><b><i>Random Forest Classification</i></b></h2>    

* Random Forest Classification is an ensemble learning method for classification
* To understand this classification, we should be aware of Bagging concept.
* <b>Bagging : Bagging is also known as Bootstrap Aggregation. The idea of bagging is to create several subsets of data
  from training sample choosen randomly with replacement. Each bag(subset) is said to be trained and form a decision tree. 
  Such that n bags will have ntrees. When a new record is passed to model to predict then that record is passed on n trees
  and the majority voting is followed to conclude the class of new record.
  * When a bag is created, it is sure that it is not including all features in the bag. Let us think in this way...
  * There are 5 independent features(f1, f2, f3, f4, f5) and when a decision tree is constructed for this bag, the head node 
    should be decided and it randomly selects between f3 and f5 and the child nodes are between f1 and f4, here f2 is left out.
  * By picking f3 or f5 it calculates the gini impurity again and starts constructing a tree.</b>
 <img src='https://prachimjoshi.files.wordpress.com/2015/07/screen_shot_2010-12-03_at_5-46-21_pm.png'>
 
  * Advantage of using Random Forest Classifier is it handles higher dimensionality data very well

<h3><b><i>Grid Search in Python
 
* Improving model's performance
* Finding optimal values for High parameters
* First Type of parameters are the parameters that are learned through the Machine Learnng Algorithm
* Second Type are the parameters that we choose
* Using <b>Grid Search CV</b> we can know which parameter to select when I make a Machine learning Model
* What is the optimal values of hyper parameters and grid search will give an answer to that because it will find optimal values for
  these parameters
