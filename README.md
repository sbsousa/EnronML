# Enron Machine Learning Algorithm

This project is an analysis of Enron persons of interest (POIs) using Machine Learning algorithms.  The project was created for Udacity's Data Analyst Nanodegree. 
 
Access the final report here: https://sbsousa.github.io/EnronML

## Project Description:

Per Udacity, the goal of this project is to "play detective and put your machine learning skills to use by building an algorithm to identify Enron Employees who may have committed fraud based on the public Enron financial and email dataset."

## Approach

First, I performed a thorough Exploratory Data Analysis to gain a better understanding of the data. Next, I shaped the data and removed outliers. Then, I used SelectKBest to determine the best features for the machine learning algorithms. Finally, I created Naive Bayes and Decision Tree algorithms to process the data.

The project was created in Python and a Jupyter Notebook. Multiple Python packages were used including scikit-learn, Pandas, NumPy, Matplotlib, and Seaborn. The final Jupyter report is provided in HTML format.

## Instructions
The Udacity files were modified to work with Python 3.9 and current packages. If you attempt to use these files, they may not work unless you recreate my environment using the packages in requirements.txt

* poi_id.py: creates the pickle (pkl) files
* tester.py: validates the selected machine learning algorithms against the pkl files and returns metrics (Accuracy, Precision, Recall, and F1) 

## License:

This project is publicly available for educational purposes. Please acknowledge this source if you use it.

## Sources

The Python scripts were provided by Udacity:

https://www.udacity.com/course/data-analyst-nanodegree--nd002

Additional sources are acknowledged in the code and report.
