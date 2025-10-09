# Lab 5: Feedforward Neural Network
#### CSC 4601 Theory of Machine Learning

![Levels, Jerry! Levels!](Images/levels.gif)

## Overview
In your previous courses, you were exposed to basic neural networks (NNs).  Early forms of NNs were developed in the 1950's but have seen a resurgence of popularity in the last decade.  The popularity is fueled by massive increases in data and compute power as well as new activation functions and optimization methods that solve the so-called "vanishing gradient" problem to enable the construction of large networks that can outperform classical machine learning models on a variety of tasks.

Neurons are organized into layers.  Each neuron implements a linear model whose output is processed through an "activation" function:

$y = f(B_{0} + B_{1}x_{1} + B_{2}x_{2} + … B_{n}x_{n})$

The hidden layers of a network often use a rectifier activation function:

$f(x) = max(0, x)$

while the output layer often uses a sigmoid function and essentially consists of logistic regression models.  In this lab, you're going to apply your knowledge about decision boundaries of linear classifiers to explore how neural networks perform classification.  We'll use a multilayer perceptron model, which is a type of dense, feedforward neural network.

For this lab, we are ommitting a train/test split - not because you shouldn't do train/test splits - but because it does not contribute to the objectives that this lab is meant to illustrate.

## Instructions

### Experimental Setup.

Import the necessary libraries and modules and then set the numpy.random.seed value to 42.
  
### Experiment 1: Data Loading, Data Visualization, and Baseline Models.
        
  1. Load the provided "xor" data set using numpy.
  
  2. Plot the data set in feature space. Make sure to label axis and observations by class.
  
  3. Train a logistic regression model [SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html) on the data set.

  4. Calculate the true positive rate (TPR) and false positive rate (FPR) on the predictions and plot the results in an ROC. The ROC plot should include a line of identity (baseline model). Additionally, includethe AUC for both models in the legend. 

### Experiment 2: Training and Visualizing Layers of a MultiLayer Perceptron (MLP)
  
  1. Train an MLP on the "xor" data set using 4 neurons:

  `mlp = MLPClassifier(hidden_layer_sizes=(4,),max_iter=1000, solver="lbfgs")`


#### Visualize Decision Boundaries Learned by Individual Neurons

  1. Extract the weight vectors for the hidden layers:

  ` mlp_models = np.vstack([mlp.intercepts_[0], mlp.coefs_[0]]).T `

  The columns of this matrix correspond to B_{0}, B_{1}, and B_{2}.  Each row corresponds to the model from a separate neuron.

  2. Re-arrange the following equation to solve for $x_{2}$:
  
  $0 = B_{0} + B_{1} x_{1} + B_{2} x_{2}$

  3. Make one figure that visualizes the features from scaled_X along with the decision boundaries for the 4 neurons in the hidden layer. Use appropriate axis limits for this visualization.

#### Visualize Decision Boundaries Resulting from Planes and ReLU Activation Function
  
  1. Create a mesh grid in the range of [-2, 2] along each dimension.

  2. Plot the grid of synthetic points as a scatter plot.
  
  3. Use the Input and Neuron classes in the provided neurons.py file to calculate the value for the first hidden layer neuron at each grid point.  (Pass the grid points into the predict() method as X).

  `input = Input()`
  
  `p_layer = Neuron([input], mlp_petals_models[0, :])`
  
  `pred = p_layer.predict(X)`

  4. Plot the model outputs as a heatmap or contourf plot.
  
  5. Repeat for the remaining 3 neurons in the hidden layer.
  

### Experiment 3: Train a Logistic Regression Model on these New Features
  
  1. Use the Input, Neuron, and HStack classes with the weights from the MLP model to recreate the hidden layer.

  `input = Input()
  
  `p_layer_1 = Neuron([input], mlp_models[0, :])`
  
  `p_layer_2 = Neuron([input], mlp_models[1, :])`
  
  `p_layer_3 = Neuron([input], mlp_models[2, :])`
  
  `p_layer_4 = Neuron([input], mlp_models[3, :])`
  
  `stacked = HStack([layer_1, layer_2, layer_3, layer_4])`

  2. Predict the transformed values to create a transformed feature matrix

  `transformed_X = stacked.predict(scaled_X)`
  
  3. Train a Logistic Regression model using the new features that were created by the MLP.
  
  4. Calculate the true positive rate (TPR) and false positive rate (FPR) on the predictions and plot the results in an ROC. The ROC plot should include the results from Experiment 1 in addition to the newly trained model. 

## Reflection Questions

Put answers to the following reflection questions at the top of your notebook (after your title and name).

### Problem 1:
  
  1. What do the parameters to the MLPClassifier class mean?
  
  2. Draw a graph of the network configuration/achetecture.
  
  3. What activation functions are used for each node?

### Problem 2:
  
  1. What are the dimensions of mlp.coefs_[0] and mlp.intercepts_[0]?  Where do those dimensions come from?
  
  2. Comment on the abilities of the lines to separate Class 1 from Class 0.

### Problem 3:
  
  1. How does a ReLU function differ from a logistic function? What would the heatmaps/contour plots look like if we used logistic function as an activation layer instead?
  
  2. A neural network consists of different layers and a final classification layer. Which activation function (ReLU or logistic) is more suitable to use for a classification layer? Which activation function is more suitable to use for an inner layer? – You may need to look up this information.

### Problem 4:
  
  1. How do the ROC's two models compare?  Did the transformed features produce a more accurate model?

  2. Are MLPs examples of linear or non-linear models?
