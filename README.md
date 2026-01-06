# AutoJudge(Predicting Programming Problem Difficulty)
AutoJudge helps in predicting the difficulty of problem with the difficulty score as well. The difficulty level is classified into three categories easy, medium and hard and score ranges from 1 to 10 and this prediction is based on the historical data provided to us( in which the rating of easiest problem is 1.1 and of hardest is 9.7) and this dataset was collected with the help of web scraping from various coding platforms like codeforces,codechef,kattis.

# Dataset
I used the provided dataset(https://github.com/AREEG94FAHAD/TaskComplexityEval-24/blob/main/problems_data.jsonl) which has 4112 samples and it initially had 8 features and the features are as follows:
1) Title
2) Description
3) input_description
4) output_description
5) sample input
6) problem class
7) problem score
8) URL of the problem
Description has 81 null values, input_description has 120 null values and output_description has 131 null values and it has 1 duplicate value (the last row with index number 4111 is duplicate of row with index number 1233 or vice-versa) and the features are not correlated because most of them are having texts.

# Approach
**Data Preprocessing:**
1) I firstly converted json file to csv becuase I am more comfortable with csv and then I found if there exist any null values or duplicate values.
2) As there were some null values as mentioned above I replaced them with " " and I tried to drop the duplicate row but it reduced the value of R2 score for regression and decreased the classification accuracy so     I did not execute this
3) I combined all the text columns and I also tried to drop url column but it also reduced the performance of metrics like R2 score and classification accuracy and then I cleaned the text.

**Feature Engineering:**

<img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/5b878958-b55f-4754-8dd0-ac973a5ca932" />

1) text_length
2) count of mathematical symbols
3) find number of sentences and the length of those sentences
4) used various constraints like memory,power of 10, memory limt..
5) io_complexity which gives the idea that how much time it was taken to read or understand that text
6) find out the number of times examples word is used because more the number of examples then there is high probability that it will be tough as compared to         others
7) I founded that how many times a keyword (which is generally used in programming problems) repeated(keyword_count) as it helps to understand difficulty of the      problem because if its' count is higher then generally its difficulty gets increased and I gave weights to them on the basis of their capability how they make     problem tough on the scale of 1-5 (keyword_weighted_score)
8) And lastly I find the number of times different keyword appeared (keyword_diversity)
9) I used TF-IDF vectors because as this dataset is made up of texts and ML can't understand text so it helps to convert into scores which is understandable, it      helps in a way that TF represents how often a word appears and IDF tells how rare that word was in a document and I took the ngram range (1,2) so that single      and double words are surely considered

And then finally I scaled and then combined all the numeric_features and tf-idf

# Models used
As I need to predict class(classification) and score(regression) both so I started with making various scatterplots so as to guess about the model which can be used so I started with the models given in the document and my training data is 80% and testing data is 20%:
**For regression:**
1) It was actually clear from my scatterplot that I should not use Linear Regression but I still used it and my R2_score was -6.57 and RMSE was 6.041
2) Then I had two Gradient Boosting and Random Forest and I knew that Random Forest will not perform much better because it won't be able to deal with this much number of tf-idf features and decision trees will get learned independently hence its metrics were not the best, R2_score was 0.129 and RMSE was 2.049
3) So I finally tried with Gradient Boosting to be precise Hist gradient boosting it was truly the best model to be used among all three and hence it's R2_score was 0.175 and RMSE was 1.99
**For classification:**
1) From scatter plot again it was intuitive that I should not use logistic regression because the data does not have linear relation and hence SVM because we know that Logistic regression and SVM share a similar baseline formulation
2) Accuracy for logistic regression wis 50.42% and confusion matrix is

   [[43  51  42]

   [19 297 109]

   [21 166  75]] 
4) Accuracy for SVM is 47.14% and confusion matrix is

   [[63  36  37]

   [52 239 134]

   [47 129  86]]
6) Accuracy for Random Forest classifier is 57.35% and confusion matrix is

   [[49  65  22]

   [27 365  33]

   [23 181  58]]
   
**Comparison Table:**
Regression Comparison:                                                        Classification Comparison:
                  Model  R2 Score  RMSE                                                             Model  Accuracy
  HistGradientBoosting  0.175903   1.99                                                     Random Forest    57.35%
         Random Forest  0.124543   2.049                                              Logistic Regression    50.42%
     Linear Regression -6.602793   6.041                                                       Linear SVM    47.14%

# Steps to run project locally 
In my main code which is model_code.py, first step was to generate pkl files( tfidf.pkl, scaler.pkl, reg_model.pkl, clf_model.pkl, label_encoder.pkl) and then there is need to go to file named app.py and then in the terminal there is need to write python -m streamlit run app.py (I used python in start because I was having an issue of location of file) then my a popup window will appear and then there I need to go to deploy section where when I will click deploy button, I firstly need to connect my github account with streamlit cloud, so I then went to site with URL(https://share.streamlit.io/) and then go to option create app and then I deployed through my github and then entered all the details but in the start I was having issues with the version of python and scikit learn hence, I created a file named requirements.txt and runtime.txt and then my app got deployed and I also tested through some of problems provided in the dataset
**App URL:** https://autojudge-predicting-programming-problem-difficulty-earbijshou.streamlit.app/

# Web interface explanation 
   I gave the title and caption to the app and then set the dimesnions and position of the boxes,I considered all those features which I considered in the main_code.py so as to maintain the uniformity and to keep the code running smoothly without any error 

   User Input
   ↓
Text Preprocessing
   ↓
TF-IDF Vectorization
   ↓
Numeric Feature Engineering
   ↓
Feature Scaling
   ↓
Feature Fusion
   ↓
Regression & Classification Inference
   ↓
Difficulty Score and Class Output

# Author Details
Name: Laksh Alawadhi
Enrollment no.: 24113076
Branch: Civil Engineering
