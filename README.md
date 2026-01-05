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
(a) text_length
(b) count of mathematical symbols
(c) find number of sentences and the length of those sentences
(d) used various constraints like memory,power of 10, memory limt..
(e) io_complexity which gives the idea that how much time it was taken to read or understand that text
(f) find out the number of times examples word is used because more the number of examples then there is high probability that it will be tough as compared to         others
(g) I founded that how many times a keyword (which is generally used in programming problems) repeated(keyword_count) as it helps to understand difficulty of the      problem because if its' count is higher then generally its difficulty gets increased and I gave weights to them on the basis of their capability how they make     problem tough on the scale of 1-5 (keyword_weighted_score)
(h) And lastly I find the number of times different keyword appeared (keyword_diversity)
(i) I used TF-IDF vectors because as this dataset is made up of texts and ML can't understand text so it helps to convert into scores which is understandable, it      helps in a way that TF represents how often a word appears and IDF tells how rare that word was in a document and I took the ngram range (1,2) so that single      and double words are surely considered
And then finally I scaled and then combined all the numeric_features and tf-idf
