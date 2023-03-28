import numpy as np
from scipy.stats import norm

def likelihood_first_derivative(x, z, theta):

  l_1 = z - np.sum(x*np.exp(theta*x))

  return l_1

def likelihood_second_derivative(x, z, theta):

  l_11 = - np.sum(x**2*np.exp(theta*x))

  return l_11

def newton_raphson_algorithm(x, z, theta_0=1):
  ## initialize
  thetas = []

  ## iteration 0
  t = 0
  theta_a = theta_0
  thetas.append(theta_a)
  l_prime = likelihood_first_derivative(x, z, theta_a)
  l_doubleprime = likelihood_second_derivative(x, z, theta_a)
  theta_b = theta_a - l_prime/l_doubleprime
  
  while np.abs(theta_a - theta_b) > 0.0000001:

    ## count iterations
    t+=1
    thetas.append(theta_b)
    theta_a = theta_b

    l_prime = likelihood_first_derivative(x, z, theta_a)
    l_doubleprime = likelihood_second_derivative(x, z, theta_a)
    theta_b = theta_a - l_prime/l_doubleprime
  
  return theta_b, t