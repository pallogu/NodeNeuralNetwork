var NeuralNetwork = require('../neural_network');
var nn = new NeuralNetwork();

var trainingSetX = [
    [0,0],
    [0,1],
    [1,0],
    [1,1]
];

var trainingSetY = [
    [0],
    [1],
    [1],
    [0]
];

var setup = {
    trainingSetX: trainingSetX,
    trainingSetY: trainingSetY,
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
        console.log('probability that y would be positive', probability);
    });
});