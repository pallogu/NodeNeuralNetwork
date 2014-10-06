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
    numberOfNodes: 2,
    numberOfExamplesPerNode: 2,
    verboseMode: true,
    lambda:0.0000001,
    learningRate: 1,
    maxCostError: 0.001,
    maxNoOfIterations: 1000000
}

nn.train(setup, function (err, model) {
    var predictionCounter = 4;

    var updatePredictionCounter = function () {
        predictionCounter--;
        if(predictionCounter === 0) {
            nn.exit();
        }
    }

    var predictionOptions = {
        numberOfActivationUnitsL1: 4,
        numberOfActivationUnitsL2: 4,
        model: model
    };

    predictionOptions.inputVector = [0,0];
    nn.predict(predictionOptions, function (err, probability){
        assert.equal(probability < 0.01, true);
        console.log('probability that [0,0] would be positive', probability);
        updatePredictionCounter();
    });

    predictionOptions.inputVector = [0,1];
    nn.predict(predictionOptions, function (err, probability){
        assert.equal(probability > 0.99, true);
        console.log('probability that [0,1] would be positive', probability);
        updatePredictionCounter();
    });

    predictionOptions.inputVector = [1,0];
    nn.predict(predictionOptions, function (err, probability){
        assert.equal(probability > 0.99, true);
        console.log('probability that [1,0] would be positive', probability);
        updatePredictionCounter();
    });

    predictionOptions.inputVector = [1,1];
    nn.predict(predictionOptions, function (err, probability){
        assert.equal(probability < 0.01, true);
        console.log('probability that [1,1] would be positive', probability);
        updatePredictionCounter();
    });
});