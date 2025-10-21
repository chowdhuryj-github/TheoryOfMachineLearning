# Problem Set: Support Vector Machines

You may complete the following entirely in a Jupyter notebook. Ensure that the notebook has your name on it. Save the notebook as a PDF and submit the PDF through Canvas.

## Problem 1: Exploring the Penalty Term Hyperparameter

Evaluating impact of the C parameter on a SVM model.

1) Load in the wine data set from scikit learn.
2) Relabel the response variable so that class 2 becomes class 1 and classes 0 and 1 become class 0. You saw something similar in Lab 2 with the Setosa/Not Setosa classes in the Iris dataset.
3) Train SVM models with a linear kernel and C values of [1,10,100,1000] on the hue and flavanoids features using the [```SVC```](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) class. (Features 10 and 6, respectively.)
4) Plot the decision boundary, margins, and support vectors for each model. Describe how the decision boundary, margins, and support vectors change with the different values of C.

## Problem 2: The Problem with Kernels

1) 1-d example

    1) Load in the dataset ``kernel_problem_1.csv``. Rows are observations,
the first column refer to $x_{1}$ features, and the second column
is given class ($y$).
    2) Plot the observations (note: we have seen observations described by
2 features so far, but that is not always the case. Observations could
be described by a single feature). Are these features linearly separable?
    3) Give the hyperplane that will divide this space by the given classes.

2) 1-d to 2-d example
   
    1) Load in the dataset ``kernel_problem_2.csv``. Rows are observations,
the first column refer to $x_{1}$ features, and the second columnis given class ($y$).
    2) Plot the observations (note: we have seen observations described by
2 features so far, but that is not always the case. Observations could
be described by a single feature). Are these features linearly separable?
    3) Use the following function to describe these observations in 2-dimensional
space: $x_{2}=x_{1}^{2}$. Plot the observations in 2-dimensional
space. Are these same observations now linearly separable?
    4) In this higher dimensional space, use the following description of
a hyperplane and a threshold to classify the points.
      - ![weight and point](Images/wandp.png)
      - ![class definition](Images/classdef.png)

3) 2-d to 3-d example

    1) Load in the dataset ``kernel_problem_3.csv``. Rows are observations,
the first and second column refer to features, the third column is
given class (y).
    2) Plot the observations. Are these observations linearly separable?
    3) Use the following function to describe these observations in 3-dimensional
space:
     - ![phi of x](Images/phi.png)
   
    4) Plot the observations from 3 separate perspectives - feature1
x feature2 , feature2 x feature3 , feature1 x feature3. Are these
same observations now linearly separable?
