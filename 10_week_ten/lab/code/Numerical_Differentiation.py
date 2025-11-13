import numpy as np

class NumericalDifferentiation:
    
    """ 
    This class should implement numerical differentiation. These methods should be able to solve a 1-dimensonal numerical 
    differentiation or n-dimensional numerical differentiation (gradient).
    """


    def __init__(self, delta):
        """
        The constructor should take the delta parameter (h) as an argument and set it as an object variable.

        There are two options for how to handle delta. Delta should either be a scalar (which you may have to later vectorize) 
        or it can be set as a k-dimensional vector in which case the the object will only be able to solve for k-dimensional gradients.
        """

        self.delta = delta
        
        

    def gradient(self, cost_func, params):
        """
        This method will use the given forumal for numerically estimating the gradient. The gradient of the cost_func should only be 
        estimated at the given params points. For example:
            - If you have a cost_func that is dependent on 2 parameters. The params vector should be shape (2,).

        The cost_func argument must follow the cost function API established in Lab06.
        This method should be robust to handle both scalar (1-dimensional) cost_func/params and any size (n-dimensional) cost_func/params.
        You are permitted to use a loop.

        This method will then return a gradient that has the same shape as params
        """

        # preparing the gradient method
        params = np.array(params)
        grad = np.zeros_like(params)
        delta = np.array(self.delta)

        if delta.size == 1:
            delta = np.full_like(params, delta.item())


        # estimating each partial derivative
        for i in range(len(params)):

            # (x+h) and (x-h)
            params_plus = np.copy(params)
            params_minus = np.copy(params)
            params_plus[i] += delta[i]
            params_minus[i] -= delta[i]

            f_plus = cost_func.cost(params_plus)
            f_minus = cost_func.cost(params_minus)

            # compute the gradient array
            grad[i] = (f_plus - f_minus) / (2 * delta[i])

        return grad
        

