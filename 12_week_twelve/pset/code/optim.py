
# importing libraries
import numpy as np


class Optimizer:

    """
    Implements Gradient Descent using numerical differentiation for calculating the gradient.
    """

    def __init__(self, step_size, max_iter, tol, delta):

        """
        Max_iter -- maximum number of iterations to run
        step_size -- also known as lambda
        tol -- Stopping parameter for difference between parameters between update steps.
        delta -- perturbation to use in numerical differentiation
        """

        self.step_size = step_size
        self.max_iter = max_iter
        self.tol = tol
        self.delta = delta
        
    
    def optimize(self, cost_func, starting_params):

        """
        Finds parameters that optimize the given cost function.
        
        This method should implement your iterative algorithm for updating your parameter estimates.
        Use an updated estimate of the gradient to update the parametes.
        
        Give consideration for what the exit conditions of this loop should be.
        
        Returns a tuple of (optimized_param, iters)
        """

        # the current paramters to use
        current_params = starting_params

        # for loop until we run out of iterations or find the minimum early
        for i in range(self.max_iter):

            # gradient tells us how steep it is
            gradient = self._gradient(cost_func, current_params)

            # new = old - step_size x gradient
            new = self._update(current_params, gradient)

            # distance between old and new parameters
            change = self._calculate_change(current_params, new)

            # if step is smaller than tolerance, we stop
            if change < self.tol:
                return new, i+1
            
            # update for the new loop
            current_params = new
        
        return current_params, self.max_iter


        
    
    def _calculate_change(self, old, new):

        """
        Calculates the change between the old and new parameters.
        Returns a scalar.
        """

        # compute distance between two parameters vectors
        return np.linalg.norm(new - old)
        
    
    def _gradient(self, cost_func, params):

        """
        Numerically estimates the gradient (first derivative) of the cost function
        at param.
        
        First-order numerical differentiation
        df/dx = [ f(x + delta) - f(x) ] / delta
        
        Should return the gradient at the caluclated point
        """

        # convert params to numpy array
        params = np.array(params, dtype=float)

        # cost at original point
        # f_x = f(params)
        f_x = cost_func.cost(params)

        # empty gradient vector -> same shape as params
        gradient = np.zeros_like(params)

        # partial derivative for each parameter
        for i in range(len(params)):
            
            new_params = params.copy()

            # add a δ to a parameter
            new_params[i] += self.delta

            # f_x = f(params + δ)
            f_x_delta = cost_func.cost(new_params)

            # forward difference
            # f(params + δ) - f(params) / δ
            gradient[i] =  (f_x_delta - f_x) / self.delta
        
        # return list of gradients
        return gradient


        
            
    def _update(self, param, gradient):

        """
        Updates the param vector using the Gradient Descent algorithm.                
        
        Returns the new parameters.  (Do not modify input)
        """

        param = np.array(param, dtype=float)
        
        # gradient descent update rule
        # new_param = old_param − λ ⋅ gradient
        new_param = param - self.step_size * gradient 
        
        return new_param
        