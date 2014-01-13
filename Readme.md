#Nodejs Neural Network
##Description
This is a nodejs implementation of dense neural network with two hidden layer and one category output layer.

It uses [compute cluster](https://github.com/lloyd/node-compute-cluster) for splitting the work into multiple cores and [numeric](http://www.numericjs.com/documentation.html) for linear algebra computations - basically verctorized implementation.

For cost function optimisation, one can use batch/mini-batches/stochastic gradient descent.

For mini-batches and stochastic gradient descent it randomises training examples using (knuth-shuffle)[https://github.com/coolaj86/knuth-shuffle]

##Instalation

Install from commnad line

`$ npm install neural_network`

##Examples



