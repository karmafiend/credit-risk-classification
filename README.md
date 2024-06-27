# credit-risk-classification
Repo for Data Analytics Module #20 Machine Learning Challenge

# Credit Risk Analysis Overview & Report:

# Analysis Purpose: 

Build, train and evaluate a supervised machine learning model based on loan applicant risk. The dataset consists of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers.

# Financial Information Definition: 

The historical dataset consisted of loan applicant historical borrowing data including previous loan size, # of loans, loan status, any derogatory marks indicating potential repayment issues, total debt, applicant and debt to income ratios. All information typically used to evaluate borrower loan suitability. Most importantly, the loan status (0=fully paid, 1=charged off) indicated applicant likelihood to repay any new loans based on past behavior. 

# Variables Overview:

The loan_status target variable was used to predict "loan_status" when evaluating loan applicants, "value_counts" (either loan status of 0 or 1) was used to classify predictions into applicants who would fully repay loans (0=fully paid) or not (1=charged off) for future loans provided by the bank.

# Model Development Process:

This was a multi-step analysis process to build, evaluate and use the supervised machine learning model, specifically:
	* Load Python / machine learning dependences: Tap into relevant analysis code libraries
	* Load the "lending_data.csv" historical borrower data set and build a Pandas dataframe to preview dataset content (specifically to identify the labels and target variable for analysis)
	* Separate the labels (i.e. "loan_status" target variable) from the features (the remaining dataset columns ex: loan_status, debt to income, etc.)
	* Split the data into training and testing datasets in order to build & run the model testing it for predictive accuracy. Specific reasons for doing so include: 
		* Providing a training set of data to be tested against a set apart test data as the basis for building the Logistic Regression analysis model
		* Doing so prevents overfitting of the data which negates the predictive power of the Logistic Regression model. If the model performs better on the training set than the test set then there is the 			potential for overfitting within the model created
		* The test set helps in assessing the bias-variance tradeoff. A model with high bias might underfit the data, while a model with high variance might overfit
	* Model Training: Created a Logistic Regression Model enabling predictive analysis of loan risk potential for borrowers
		* Fitted the logistic regression model using the training data
		* Saved the predictions on the test data labels using testing feature data and the fitted model; scored the model; made predictions
	* Prediction and evaluation: Then evaluated the Logistic Regression model's performance by creating a Confusion Matrix and generating an accuracy score
		* Displayed the Confusion Matrix and Accuracy Scores for model performance analysis

# Notes on Methods Used: 

For this credit risk analysis a tried and true Logistic Regression machine learning model was used for a variety of reasons, but especially due to its ability to output loan default predictions with a high degree of accuracy. Logistic Regression models are widely used in the financial industry for binary analyses such as this credit risk where the primary need was to predict whether borrower would either pay off a loan (0=fully paid) or not (1=charged off). A number of other model benefits that support the use of the Logistic Regression model include its computational efficiency (i.e. use of computing resources to generate and use models for analysis), and the ease by which its results can be interpreted by loan officers and other bank officials.

# Credit Risk Analysis Results:

* Credit Risk Logistic Regression Machine Learning Model 1: Class 0, "fully paid" borrowers
	* Precision Score: 1.0. Score of 1.0 indicates that the model is accurate 100% of the 		time in predicting those applicants who will not default on their loans.  
	* Recall Score: 0.99. Similarly, the model is stronger at predicting Class 0, non-		default borrowers than those that do default with its recall score of 0.99 indicating it 	correctly predicts those borrowers will not default 99% of the time 

* Credit Risk Logistic Regression Machine Learning Model 2: Class 1, "charged off" borrowers
	* Precision Score: 0.85. The 0.85 precision score for the default borrowers indicates an 	85% success rate at predicting loan results - a much weaker score than Class 0 which 		could potentially create more loan risk for any financial institution lending money to a 	borrower since it's not as precise at predicting those who will default on their loans
	* Recall Score: 0.91. The model also scores lower here on its recall score with its 0.91 	score indicating that it predicts those that default 91% of the time

# Credit Risk Analysis Summary:
Overall the macro and weighted averages for both classes indicate the model is balanced at predicting loan performance by future borrowers taken as a whole across both classes of borrowers as indicated by the 99% accuracy score
	* Accuracy Score: Measures how often is the model correct in interpreting and predicting 	results (in this case, how likely are loan applicants to default). With an overall 99% 		accuracy score, this model is correct 99% of the time at predicting the overall 		likelihood a loan will be repaid

However, when examined more closely by classes, 0=fully paid and 1=charged off borrowers, the model's lower accuracy levels at predicting those who will default (Class 1=charged off) could introduce significant risk for a financial institution since the model struggles to predict bad loans at the same accuracy level it does for those who will repay their loans. Use of the model should thus be used judiciously and most likely compared with other models, ones that may be more accurate at predicting loan defaults
