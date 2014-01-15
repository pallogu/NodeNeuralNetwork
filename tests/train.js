var assert = require('assert');
var NeuralNetwork = require('../neural_network');
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
    verboseMode: true,
    learningRate: 0.5,
    maxCostError: 0.001,
    maxNoOfIterations: 100000
}

nn.train(setup, function (err, model) {
    nn.predict([0,0], function (err, probability){
        assert.equal(probability < 0.01, true);
        console.log('probability that [0,0] would be positive', probability);
    });

    nn.predict([0,1], function (err, probability){
        assert.equal(probability > 0.99, true);
        console.log('probability that [0,1] would be positive', probability);
    });

    nn.predict([1,0], function (err, probability){
        assert.equal(probability > 0.99, true);
        console.log('probability that [1,0] would be positive', probability);
    });

    nn.predict([1,1], function (err, probability){
        assert.equal(probability < 0.01, true);
        console.log('probability that [1,1] would be positive', probability);
    });
});