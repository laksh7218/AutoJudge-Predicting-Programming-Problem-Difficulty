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

