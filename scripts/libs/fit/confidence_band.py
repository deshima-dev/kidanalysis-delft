#------------------------------------------------------------
# Calculate confidence band using result from lmfit
# ref:
#  https://lmfit.github.io/lmfit-py/fitting.html#lmfit.minimizer.MinimizerResult
#  https://www.astro.rug.nl/software/kapteyn/kmpfittutorial.html#confidence-and-prediction-intervals
#  https://www.astro.rug.nl/software/kapteyn/EXAMPLES/kmpfit_example_partialdervs_confidence.py
#  http://stackoverflow.com/questions/24633664/confidence-interval-for-exponential-curve-fit/37080916#37080916
#------------------------------------------------------------
#

import lmfit
import numpy as np

def confpred_band(x, dfdp, prob, fitobj, f, prediction, abswei=False, err=None):
   #----------------------------------------------------------
   # Return values for a confidence or a prediction band.
   # See documentation for methods confidence_band and 
   # prediction_band
   #----------------------------------------------------------   
   from scipy.stats import t
   # Given the confidence or prediction probability prob = 1-alpha
   # we derive alpha = 1 - prob 
   alpha = 1 - prob
   prb = 1.0 - alpha/2
   tval = t.ppf(prb, fitobj.nfree)
   
   C = fitobj.covar
   n = len(fitobj.params)              # Number of parameters from covariance matrix
   p = fitobj.params.valuesdict()      # best fit parameter value dictionary
   N = len(x)
   if abswei:
      covscale = 1.0  # Do not apply correction with red. chi^2
   else:
      covscale = fitobj.redchi

   df2 = np.zeros(N)
   for j in range(n):
      for k in range(n):
         df2 += dfdp[j]*dfdp[k]*C[j,k]

   if prediction:
      df = np.sqrt(err*err+covscale*df2)
   else:
      df = np.sqrt(covscale*df2)

   y = f(x, p)
   delta = tval * df
   upperband = y + delta
   lowerband = y - delta 

   return y, upperband, lowerband

def confidence_band(x, dfdp, confprob, fitobj, f, err=None, abswei=False):
   #----------------------------------------------------------
   # Given a value for x, calculate the error df in y = model(p,x)
   # This function returns for each x in a NumPy array, the
   # upper and lower value of the confidence interval. 
   # The arrays with limits are returned and can be used to
   # plot confidence bands.  
   # 
   #
   # Input:
   #
   # x        NumPy array with values for which you want
   #          the confidence interval.
   #
   # dfdp     A list with derivatives. There are as many entries in
   #          this list as there are parameters in your model.
   #
   # confprob Confidence probability in percent (e.g. 90% or 95%).
   #          From this number we derive the confidence level 
   #          (e.g. 0.05). The Confidence Band
   #          is a 100*(1-alpha)% band. This implies
   #          that for a given value of x the probability that
   #          the 'true' value of f(x, p) falls within these limits is
   #          100*(1-alpha)%.
   # 
   # fitobj   The MinimizerResult object from a fit with lmfit
   #
   # f        A function that returns a value y = f(x, p)
   #          p are the best-fit parameters dictionary and x is a NumPy array
   #          with values of x for which you want the confidence interval.
   #
   # abswei   Are the weights absolute? For absolute weights we take
   #          unscaled covariance matrix elements in our calculations.
   #          For unit weighting (i.e. unweighted) and relative 
   #          weighting, we scale the covariance matrix elements with 
   #          the value of the reduced chi squared.
   #
   # Returns:
   #
   # y          The model values at x: y = f(x, p)
   # upperband  The upper confidence limits
   # lowerband  The lower confidence limits   
   #
   # Note:
   #
   # If parameters were fixed in the fit, the corresponding 
   # error is 0 and there is no contribution to the confidence
   # interval.
   #----------------------------------------------------------   

   return confpred_band(x, dfdp, confprob, fitobj, f, prediction=False, err=err, abswei=abswei)


def prediction_band(x, dfdp, predprob, fitobj, f, err=None, abswei=False):
   #----------------------------------------------------------
   # Given a value for x, calculate the error df in y = model(p,x)
   # This function returns for each x in a NumPy array, the
   # upper and lower value of the prediction interval. 
   # The arrays with limits are returned and can be used to
   # plot confidence bands.  
   # 
   #
   # Input:
   #
   # x        NumPy array with values for which you want
   #          the prediction interval.
   #
   # dfdp     A list with derivatives. There are as many entries in
   #          this list as there are parameters in your model.
   #
   # predprob Prediction probability in percent (e.g. 0.9 or 0.95).
   #          From this number we derive the prediction level 
   #          (e.g. 0.05). The Prediction Band
   #          is a 100*(1-alpha)% band. This implies
   #          that values of one or more future observations from
   #          the same population from which a given data set was sampled,
   #          will fall in this band with a probability of 100*(1-alpha)%
   # 
   # fitobj   The MinimizerResult object from a fit with lmfit
   #
   # f        A function that returns a value y = f(x, p)
   #          p are the best-fit parameters dictionary and x is a NumPy array
   #          with values of x for which you want the confidence interval.
   #
   # abswei   Are the weights absolute? For absolute weights we take
   #          unscaled covariance matrix elements in our calculations.
   #          For unit weighting (i.e. unweighted) and relative 
   #          weighting, we scale the covariance matrix elements with 
   #          the value of the reduced chi squared.
   #
   # Returns:
   #
   # y          The model values at x: y = f(x, p)
   # upperband  The upper prediction limits
   # lowerband  The lower prediction limits   
   #
   # Note:
   #
   # If parameters were fixed in the fit, the corresponding 
   # error is 0 and there is no contribution to the prediction
   # interval.
   #----------------------------------------------------------   

   return confpred_band(x, dfdp, predprob, fitobj, f, prediction=True, err=err, abswei=abswei)

