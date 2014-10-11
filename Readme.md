#Nodejs Neural Network
##Description
This is a nodejs implementation of dense neural network with two hidden layer and one category output layer.

It uses [compute cluster](https://github.com/lloyd/node-compute-cluster) for splitting the work into multiple cores.

For cost function optimisation, one can use batch/mini-batches/stochastic gradient descent.

For mini-batches and stochastic gradient descent it randomises training examples using (knuth-shuffle)[https://github.com/coolaj86/knuth-shuffle]

##Instalation

Install from command line

`$ npm install neural_network`

##Examples

	var NeuralNetwork = require('neural_network');
	var nn = new NeuralNetwork();
	
	var trainingSetInput = [
	    [0,0],
	    [0,1],
	    [1,0],
	    [1,1]
	];
	
	var trainingSetOutput = [
	    [0],
	    [1],
    	[1],
	    [0]
	];
	
	var setup = {
	    trainingSetInput: trainingSetInput,
	    trainingSetOutput: trainingSetOutput,
	    numberOfActivationUnitsL1: 4,
	    numberOfActivationUnitsL2: 4,
	    numberOfNodes: 1,
	    numberOfExamplesPerNode: 4,
	    learningRate: 0.5,
	    maxCostError: 0.001,
	    maxNoOfIterations: 100000
	}
	
	nn.train(setup, function (err, model) {
	    nn.predict([1,1], function (err, probability){
	        console.log('probability that y would be 	positive', probability);
	        nn.exit();
	    });
	});

## Setup

Setup takes following required parameters

* trainingSetX: Please use matrix representation
* trainingSetY: Please use matrix representation
* numberOfActivationUnitsL1: Number of activation units in first hidden layer
* numberOfActivationUnitsL2: Number of activation units in second hidden layer

Following parameters are optional

* numberOfNodes: (int) Used for map reduce
* numberOfExamplesPerNode: (int)
* learningRate: (number) This number is used in gradient descent
* lambda: (number) regularisation parameter
* maxCostError: (number) This parameter is used to stop training. If the value of cost function is less than maxCostError callback will be called
* maxGradientSize: (number) This parameter is used also to stop training. If the gradient size is smaller than this value, there is a check of whether secondary derivations are positive (diagonal of Hessian)
* maxNoOfIterations: (int) This parameter is used to stop training. The default value is Number.MAX_VALUE
* model: (array) This is the starting point for gradient descent optimisation. If you do not provide this one will be randomly generated. This is useful if you already have a model and want to adjust it by new examples. The *train* method calls the callback with trained model as parameter.
* verboseMode: (boolean) If set to true it will report the progress of learning



##Optimisation

### Stochastic gradient descent
Set:
	
	numberOfNodes = 1
	numberOfExamplesPerNode = 1
	
### Batch gradient descent

Set:

	var os = require(os);
	
	numberOfNodes = os.cpus().length - 1;
	numberOfExamples = Math.floor(trainingSetX.length / numberOfNodes);
	
### Mini batch gradient descent

Set for example:

	var os = require(os);
	
	numberOfNodes = os.cpus().length - 1;
	numberOfExamples = 10;


## Some implementation notes

In other words you have to find a balance between numberOfExamplesPerNode and numberOfNodes for mini batch gradient descent.

## Copyright

Copyright (c) 2014, Paul Gustafik paul.gustafik@gmail.com

Permission to use, copy, modify, and/or distribute this software for any purpose with or without fee is hereby granted, provided that the above copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

