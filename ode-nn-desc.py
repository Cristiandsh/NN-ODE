# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 16:49:28 2018

Sigo como referncia el ejemplo de ODE primer orden de: https://becominghuman.ai/neural-networks-for-solving-differential-equations-fa230ac5e04c 

Este a su vez se basa bastante en https://arxiv.org/pdf/physics/9705023.pdf 

@author: 2017
"""

import autograd.numpy as np
from autograd import grad
import autograd.numpy.random as npr

from autograd.core import primitive

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
npr.seed(0)

nx = 10
dx = 1. / nx


def A(x):
	'''
        Left part of initial equation
    '''
	return x + (1. + 3. * x**2) / (1. + x + x**3)


def B(x):
	'''
        Right part of initial equation
    '''
	return x**3 + 2. * x + x**2 * ((1. + 3. * x**2) / (1. + x + x**3))


def f(x, psy):
	'''
        d(psy)/dx = f(x, psy)
        This is f() function on the right
    '''
	return B(x) - psy * A(x)


def psy_analytic(x):
	'''
        Analytical solution of current problem
    '''
	return (np.exp((-x**2) / 2.)) / (1. + x + x**3) + x**2


x_space = np.linspace(0, 1, nx)
y_space = psy_analytic(x_space)
psy_fd = np.zeros_like(y_space)
psy_fd[0] = 1.  # IC

for i in range(1, len(x_space)):
	psy_fd[i] = psy_fd[i - 1] + B(x_space[i]) * dx - psy_fd[i - 1] * A(
	    x_space[i]) * dx

plt.figure()
plt.plot(x_space, y_space)
plt.plot(x_space, psy_fd)
plt.show()


def sigmoid(x):
	return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):
	return sigmoid(x) * (1 - sigmoid(x))


def neural_network(W, x):
	a1 = sigmoid(np.dot(x, W[0]))
	return np.dot(a1, W[1])


def d_neural_network_dx(W, x, k=1):
	return np.dot(
	    np.dot(W[1].T, W[0].T**k),
	    sigmoid_grad(x))  #diferencial del resultado de la red de nuronas.


def loss_function(W, x):
	loss_sum = 0.
	for xi in x:
		net_out = neural_network(
		    W, xi)[0][0]  #valor absoluto por resultado de neurna intermedia.
		psy_t = 1. + xi * net_out  #simbolo giergo (PSY o PSI) subindice t que significa trial solution
		d_net_out = d_neural_network_dx(W, xi)[0][0]
		d_psy_t = net_out + xi * d_net_out  #Sale de deriva la trial solution || regla de derivacion de un producto
		func = f(xi, psy_t)
		err_sqr = (
		    d_psy_t - func
		)**2  #al pasar todo los terminos la funcion general con la derivada se iguala a 0. Así que el problema consiste en hacer el error lo mas cercano a 0 posible.

		loss_sum += err_sqr
	print('loss_sum = ', loss_sum)
	return loss_sum


W = [npr.randn(1, 10), npr.randn(10, 1)]
lmb = 0.001

# x = np.array(1)
# print neural_network(W, x)
# print d_neural_network_dx(W, x)

for i in range(1000):
	loss_grad = grad(loss_function)(
	    W, x_space
	)  #Se busca optimizar el problema buscnado el minimo, por eso se resta a los pesos W.

	#     print loss_grad[0].shape, W[0].shape
	#     print loss_grad[1].shape, W[1].shape

	W[0] = W[0] - lmb * loss_grad[0]
	W[1] = W[1] - lmb * loss_grad[1]
	#print(i)
	#print('loss_grad = ' ,loss_grad[0][0][1])
	#print('W = ' ,W[0][0][1])

#     print loss_function(W, x_space)
res = [1 + xi * neural_network(W, xi)[0][0] for xi in x_space]

print(W)

fig = plt.figure()
plt.plot(x_space, y_space)
plt.plot(x_space, psy_fd)
plt.plot(x_space, res)
plt.show()

fig.savefig('graph.png')
