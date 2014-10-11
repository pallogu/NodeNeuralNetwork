var _ = require('lodash');
var assert = require('assert');
var la = require('../helpers/linear_algebra.helper');

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
var initialThetaVec = la.randomVector((numberOfFeatures + 1) * numberOfActivationUnitsL1 + (numberOfActivationUnitsL1 + 1) * numberOfActivationUnitsL2 + numberOfActivationUnitsL2 + 1, 0.5);

var initialThetaVecPlusEps = [];
var initialThetaVecMinusEps = [];

var counter = initialThetaVec.length - 1;

var errorSum = 0;
var epsilon = 0.0001;

var splitThetaIntoMatrices = function (setup) {
    var ThetaVec = setup.ThetaVec;
    var numberOfFeatures = setup.numberOfFeatures;
    var numberOfActivationUnitsL1 = setup.numberOfActivationUnitsL1;
    var numberOfActivationUnitsL2 = setup.numberOfActivationUnitsL2;
    var numberOfOutputUnits = setup.numberOfOutputUnits;

    var theta1Vec = ThetaVec.slice(0, (numberOfFeatures + 1) * numberOfActivationUnitsL1);;
    var theta2Vec = ThetaVec.slice((numberOfFeatures + 1) * numberOfActivationUnitsL1, (numberOfActivationUnitsL1 + 1) * (numberOfActivationUnitsL2) + (numberOfFeatures + 1) * numberOfActivationUnitsL1);;
    var theta3Vec = ThetaVec.slice((numberOfActivationUnitsL1 + 1) * (numberOfActivationUnitsL2) + (numberOfFeatures + 1) * numberOfActivationUnitsL1, ThetaVec.length);;

    var r = 0;

    var Theta1 = new Array(numberOfActivationUnitsL1);
    var Theta2 = new Array(numberOfActivationUnitsL2);
    var Theta3 = new Array(numberOfOutputUnits);


    for(r = 0; r < numberOfActivationUnitsL1; r++) {
        Theta1[r] = theta1Vec.slice(r * (numberOfFeatures + 1), (r + 1) * (numberOfFeatures + 1))
    }

    for(r = 0; r < numberOfActivationUnitsL2; r++) {
        Theta2[r] = theta2Vec.slice(r * (numberOfActivationUnitsL1 + 1), (r + 1) * (numberOfActivationUnitsL1 + 1))
    }

    for(r = 0; r < numberOfOutputUnits; r++) {
        Theta3[r] = theta3Vec.slice(r * (numberOfActivationUnitsL2 + 1), (r + 1) * (numberOfActivationUnitsL2 + 1))
    }

    return {
        Theta1 : Theta1,
        Theta2 : Theta2,
        Theta3 : Theta3
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

    initialThetaVecPlusEps = initialThetaVec.slice(0);
    initialThetaVecMinusEps = initialThetaVec.slice(0);

    initialThetaVecMinusEps[counter] = initialThetaVecMinusEps[counter] - epsilon;
    initialThetaVecPlusEps[counter] = initialThetaVecPlusEps[counter] + epsilon;

    var thetas = splitThetaIntoMatrices({
        ThetaVec: initialThetaVec,
        numberOfActivationUnitsL1: numberOfActivationUnitsL1,
        numberOfActivationUnitsL2: numberOfActivationUnitsL2,
        numberOfFeatures: numberOfFeatures,
        numberOfOutputUnits: 1
    });

    computeCluster.enqueue({
        Theta1: thetas.Theta1,
        Theta2: thetas.Theta2,
        Theta3: thetas.Theta3,
        lambda: 1,
        X: trainingSetInput,
        Y: trainingSetOutput
    }, function (err, r) {
        var backpropVector = _.flatten(r.D1).concat(_.flatten(r.D2), _.flatten(r.D3));
        backProgGradient = backpropVector[counter] / trainingSetOutput.length;
        updateCounters();
    });

    var thetasMinusEpsilon = splitThetaIntoMatrices({
        ThetaVec: initialThetaVecMinusEps,
        numberOfActivationUnitsL1: numberOfActivationUnitsL1,
        numberOfActivationUnitsL2: numberOfActivationUnitsL2,
        numberOfFeatures: numberOfFeatures,
        numberOfOutputUnits: 1
    });

    computeCluster.enqueue({
        Theta1: thetasMinusEpsilon.Theta1,
        Theta2: thetasMinusEpsilon.Theta2,
        Theta3: thetasMinusEpsilon.Theta3,
        lambda: 1,
        X: trainingSetInput,
        Y: trainingSetOutput
    }, function (err, r) {
        costMinusEpsilon = r.cost / trainingSetOutput.length;
        updateCounters();
    });

    var thetasPlusEpsilon = splitThetaIntoMatrices({
        ThetaVec: initialThetaVecPlusEps,
        numberOfActivationUnitsL1: numberOfActivationUnitsL1,
        numberOfActivationUnitsL2: numberOfActivationUnitsL2,
        numberOfFeatures: numberOfFeatures,
        numberOfOutputUnits: 1
    });

    computeCluster.enqueue({
        Theta1: thetasPlusEpsilon.Theta1,
        Theta2: thetasPlusEpsilon.Theta2,
        Theta3: thetasPlusEpsilon.Theta3,
        lambda: 1,
        X: trainingSetInput,
        Y: trainingSetOutput
    }, function (err, r) {
        costPlusEpsilon = r.cost / trainingSetOutput.length;
        updateCounters();
    });
};

computeDifferenceBetweenNumericlPartialDerivativeAndBackprop();
