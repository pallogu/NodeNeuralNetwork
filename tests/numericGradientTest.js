var _ = require('underscore');
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

var errorSum=0;
var epsilon = 0.0001;

var computeDifferenceBetweenNumericlPartialDerivativeAndBackprop = function() {
     var bc = 3;
     var costPlusEpsilon = 0;
     var costMinusEpsilon = 0;
     var backProgGradient = 0;

     var updateCounters = function () {
         bc--;
         if (bc<1) {
             errorSum += Math.abs((costPlusEpsilon - costMinusEpsilon)/2/ epsilon - backProgGradient/4);
             counter--;
             if(counter + 1  > 0) {
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

     computeCluster.enqueue({
         numberOfFeatures: numberOfFeatures,
         numberOfActivationUnitsL1: numberOfActivationUnitsL1,
         numberOfActivationUnitsL2: numberOfActivationUnitsL2,
         ThetaVec: initialThetaVec,
         lambda: 1,
         X: trainingSetInput,
         Y: trainingSetOutput
     }, function (err, r) {
         backProgGradient = r[1][counter];
         updateCounters();
     });

     computeCluster.enqueue({
         numberOfFeatures: numberOfFeatures,
         numberOfActivationUnitsL1: numberOfActivationUnitsL1,
         numberOfActivationUnitsL2: numberOfActivationUnitsL2,
         ThetaVec: initialThetaVecMinusEps,
         lambda: 1,
         X: trainingSetInput,
         Y: trainingSetOutput
     }, function (err, r) {
         costMinusEpsilon = r[0]/trainingSetOutput.length;
         updateCounters();
     });

     computeCluster.enqueue({
         numberOfFeatures: numberOfFeatures,
         numberOfActivationUnitsL1: numberOfActivationUnitsL1,
         numberOfActivationUnitsL2: numberOfActivationUnitsL2,
         lambda: 1,
         ThetaVec: initialThetaVecPlusEps,
         X: trainingSetInput,
         Y: trainingSetOutput
     }, function (err, r) {
         costPlusEpsilon = r[0]/trainingSetOutput.length;
         updateCounters();
     });
};

computeDifferenceBetweenNumericlPartialDerivativeAndBackprop();
