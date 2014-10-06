var _ = require('lodash');
var assert = require('assert');
var numeric = require('numeric');

var path = require('path');

const ComputeCluster = require('compute-cluster');
var computeCluster = new ComputeCluster({
    module: path.join(__dirname, '..', 'helpers/neural_network.helper.js')
});

var trainingSetInput = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
];

var trainingSetOutput = [
    [0],
    [1],
    [1],
    [0]
];

var numberOfFeatures = 2;
var numberOfActivationUnitsL1 = 3;
var numberOfActivationUnitsL2 = 3;
var initialThetaVec = numeric.sub(numeric.random([1, (numberOfFeatures + 1) * numberOfActivationUnitsL1 + (numberOfActivationUnitsL1 + 1) * numberOfActivationUnitsL2 + numberOfActivationUnitsL2 + 1])[0], 0.5);

var initialThetaVecPlusEps = [];
var initialThetaVecMinusEps = [];

var counter = initialThetaVec.length - 1;

var errorSum = 0;
var epsilon = 0.0001;

var splitThetaIntoVecs = function (setup) {
    var ThetaVec = setup.ThetaVec;
    var numberOfFeatures = setup.numberOfFeatures;
    var numberOfActivationUnitsL1 = setup.numberOfActivationUnitsL1;
    var numberOfActivationUnitsL2 = setup.numberOfActivationUnitsL2;
    var numberOfOutputUnits = setup.numberOfOutputUnits;

    var theta1Vec = ThetaVec.slice(0, (numberOfFeatures + 1) * numberOfActivationUnitsL1);
    var theta2Vec = ThetaVec.slice((numberOfFeatures + 1) * numberOfActivationUnitsL1, (numberOfActivationUnitsL1 + 1) * (numberOfActivationUnitsL2) + (numberOfFeatures + 1) * numberOfActivationUnitsL1);
    var theta3Vec = ThetaVec.slice((numberOfActivationUnitsL1 + 1) * (numberOfActivationUnitsL2) + (numberOfFeatures + 1) * numberOfActivationUnitsL1, ThetaVec.length);


    return {
        Theta1: {
            vector: theta1Vec,
            rows: numberOfActivationUnitsL1,
            cols: numberOfFeatures + 1
        },
        Theta2: {
            vector: theta2Vec,
            rows: numberOfActivationUnitsL2,
            cols: numberOfActivationUnitsL1 + 1
        },
        Theta3: {
            vector: theta3Vec,
            rows: numberOfOutputUnits,
            cols: numberOfActivationUnitsL2 + 1
        }
    }
};

var computeDifferenceBetweenNumericlPartialDerivativeAndBackprop = function () {
    var bc = 3;
    var costPlusEpsilon = 0;
    var costMinusEpsilon = 0;
    var backProgGradient = 0;

    var updateCounters = function () {
        bc--;
        if (bc < 1) {
            errorSum += Math.abs((costPlusEpsilon - costMinusEpsilon) / 2 / epsilon - backProgGradient);
            counter--;
            if (counter + 1 > 0) {
                computeDifferenceBetweenNumericlPartialDerivativeAndBackprop();
            } else {
                console.log('Sum of differences between numerical and back propagation gradients: %s should be smaller than 1e-8 or such', errorSum);
                assert.equal(errorSum < 1e-8, true);
                computeCluster.exit();
            }
        }
    };

    initialThetaVecPlusEps = numeric.clone(initialThetaVec);
    initialThetaVecMinusEps = numeric.clone(initialThetaVec);

    initialThetaVecMinusEps[counter] = initialThetaVecMinusEps[counter] - epsilon;
    initialThetaVecPlusEps[counter] = initialThetaVecPlusEps[counter] + epsilon;

    var thetaVectors = splitThetaIntoVecs({
        ThetaVec: initialThetaVec,
        numberOfActivationUnitsL1: numberOfActivationUnitsL1,
        numberOfActivationUnitsL2: numberOfActivationUnitsL2,
        numberOfFeatures: numberOfFeatures,
        numberOfOutputUnits: 1
    });

    computeCluster.enqueue({
        Theta1: thetaVectors.Theta1,
        Theta2: thetaVectors.Theta2,
        Theta3: thetaVectors.Theta3,
        lambda: 1,
        X: trainingSetInput,
        Y: trainingSetOutput
    }, function (err, r) {
        var backpropVector = r.D1.concat(r.D2, r.D3);
        backProgGradient = backpropVector[counter] / trainingSetOutput.length;
        updateCounters();
    });

    var thetaVectorsMinusEpsilon = splitThetaIntoVecs({
        ThetaVec: initialThetaVecMinusEps,
        numberOfActivationUnitsL1: numberOfActivationUnitsL1,
        numberOfActivationUnitsL2: numberOfActivationUnitsL2,
        numberOfFeatures: numberOfFeatures,
        numberOfOutputUnits: 1
    });

    computeCluster.enqueue({
        Theta1: thetaVectorsMinusEpsilon.Theta1,
        Theta2: thetaVectorsMinusEpsilon.Theta2,
        Theta3: thetaVectorsMinusEpsilon.Theta3,
        lambda: 1,
        X: trainingSetInput,
        Y: trainingSetOutput
    }, function (err, r) {
        costMinusEpsilon = r.cost / trainingSetOutput.length;
        updateCounters();
    });

    var thetaVectorsPlusEpsilon = splitThetaIntoVecs({
        ThetaVec: initialThetaVecPlusEps,
        numberOfActivationUnitsL1: numberOfActivationUnitsL1,
        numberOfActivationUnitsL2: numberOfActivationUnitsL2,
        numberOfFeatures: numberOfFeatures,
        numberOfOutputUnits: 1
    });

    computeCluster.enqueue({
        Theta1: thetaVectorsPlusEpsilon.Theta1,
        Theta2: thetaVectorsPlusEpsilon.Theta2,
        Theta3: thetaVectorsPlusEpsilon.Theta3,
        lambda: 1,
        X: trainingSetInput,
        Y: trainingSetOutput
    }, function (err, r) {
        costPlusEpsilon = r.cost / trainingSetOutput.length;
        updateCounters();
    });
};

computeDifferenceBetweenNumericlPartialDerivativeAndBackprop();
