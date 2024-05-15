# wls.py

import numpy as np
from utils import Constants

class WeightAlgorithmsParameters:
    """
    Class defining the avalable parameters used weighting functions
    """
    def __init__(self, nIter, _numMeas, _cn0, _el, _az, _pRes):
        self.nIter = nIter
        self.numMeas = _numMeas
        self.cn0 = _cn0
        self.el = _el
        self.az = _az
        self.pRes = _pRes

class RobustWeightFunctions:
    """
    Class to define the Weight Functions for robust estimation
    """

    def weight_function_huber(res, alpha, isabs = False): 
        """
        Huber function following Table 1 in reference

        :param res: residuals
        :param alpha: alpha parameter for the weighting function.
        """
        if isabs:
            resx = np.abs(res)
        else:
            resx = res
        weights = np.ones(np.shape(res))
        idx = resx > alpha
        weights[idx] = alpha / res[idx]
        return weights
    

    def weight_function_tuckey(res, alpha):
        """
        Tukey bi-weight function following Table 1 in reference

        :param res: residuals
        :param alpha: alpha parameter for the weighting function.
        """

        weights = np.ones(np.shape(res)) * Constants.epsilonErrMag6
        idx = np.abs(res) <= alpha
        weights[idx] = (1 - (res[idx] / alpha)**2)**2
        return weights


class WeightAlgorithms:
    """
    Class to define the algorithms to use for weighted LS
    """

    def __init__(self, weight_function = RobustWeightFunctions.weight_function_huber):
        self.weight_function = weight_function
    
    def algo_linear(self, params):
        """
        No weight at all, returns Nx1 vector of 1s
        """
        return np.ones(params.numMeas)
    
    def algo_cn0_elev(self, params):
        """
        Based on CN0 and Elevation\n
        Following equation 8 in https://gssc.esa.int/navipedia/index.php?title=Best_Linear_Unbiased_Minimum-Variance_Estimator_(BLUE) \n
        with a = 0, b = CN0 and c = 1.

        :param params: input parameters as defined in WeightAlgorithmsParameters

        :return weights: Nx1 array of weights, where N is the number of available measurements.
        """
        cn0Norm = 10**(params.cn0/10)
        cn0Norm = cn0Norm / cn0Norm.max()
        weights = cn0Norm * np.exp(params.el)
        return weights

    def algo_m_estimation(self, params):
        """
        Robust estimation (simplification) based on IRLS (M-Estimation) from: \n

        Akram, M.A.; Liu, P.; Wang, Y.; Qian, J. "GNSS Positioning Accuracy Enhancement Based on 
        Robust Statistical MM Estimation Theory for Ground Vehicles in Challenging Environments."
        Appl. Sci. 2018, 8, 876. https://doi.org/10.3390/app8060876

        :param params: input parameters as defined in WeightAlgorithmsParameters

        :return weights: Nx1 array of weights, where N is the number of available measurements.
        """
        s0 = 1.4826 * np.median(params.pRes)
        resScaled = params.pRes / s0
        weights = self.weight_function(resScaled, Constants.alpha_efficiency_95) + Constants.epsilonErrMag10 * np.ones(resScaled.size)
        return weights
    
    def algo_cn0_elev_tuned(self, params):
        """
        Tuned weighting function based on elevation and CN0.

        :param params: input parameters as defined in WeightAlgorithmsParameters

        :return weights: Nx1 array of weights, where N is the number of available measurements.
        """
        el = params.el
        cn0 = params.cn0
        pres = params.pRes
        elp = np.maximum(np.median(params.el), np.deg2rad(70))
        cn0p = np.maximum(np.median(params.cn0), 40)
        beta = RobustWeightFunctions.weight_function_huber(cn0p - cn0, 3, isabs=False)
        beta *= RobustWeightFunctions.weight_function_huber(np.abs(pres), 0.5, isabs=True)
        weights =  beta * np.exp(-(el - elp)**2) + np.ones(params.numMeas) * Constants.epsilonErrMag6 # to avoid having all 0s and leading to singular covariance matrix
        return weights
    