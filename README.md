<h2><b><i>Simple Linear Regression using Python</i></b></h2>

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
    
    
