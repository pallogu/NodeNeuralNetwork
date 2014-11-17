'use strict';
var La = require('./linear_algebra.helper.js');

var cp = require('child_process');

var path = require('path');

var OptimisationStepExecutor = function () {};

OptimisationStepExecutor.prototype.execute = function (options, callback) {

    var thetas = options.thetas;
    var lambda = options.lambda;
    var learningRate = options.learningRate;
    var numberOfNodes = options.helpers.length;
    var numberOfExamplesPerNode = options.numberOfExamplesPerNode;
    var trainingSetInput = options.trainingSetInput;
    var trainingSetOutput = options.trainingSetOutput;
    var helpers = options.helpers;

    var la = La();

    var D1 = la.create2DMatrix(thetas.Theta1.length, thetas.Theta1[0].length);
    var D2 = la.create2DMatrix(thetas.Theta2.length, thetas.Theta2[0].length);
    var D3 = la.create2DMatrix(thetas.Theta3.length, thetas.Theta3[0].length);

    var trainingSetSliceStart = 0;
    var trainingSetSliceEnd = 0;
    var trainingSetInputSlice = [] ;
    var trainingSetOutputSlice = [];
    var trainingRegressionCounter = numberOfNodes;

    var gradientScalar = 0;
    var thetaVecScalar = 0;
    var totalCost = 0;
    var batchSize = 0;
    var k;
    var allNodesFinished = false;


    for (k = 0; k < numberOfNodes; k++) {
        helpers[k].once('message', processHelperResult);

        trainingSetSliceStart = numberOfExamplesPerNode * k;
        trainingSetSliceEnd = numberOfExamplesPerNode * (k + 1);
        trainingSetInputSlice = trainingSetInput.slice(trainingSetSliceStart, trainingSetSliceEnd);
        trainingSetOutputSlice = trainingSetOutput.slice(trainingSetSliceStart, trainingSetSliceEnd);

        batchSize += trainingSetInputSlice.length;

        var setup = {
            batchSize: batchSize,
            Theta1 : thetas.Theta1,
            Theta2 : thetas.Theta2,
            Theta3 : thetas.Theta3,
            lambda: lambda,
            X: trainingSetInputSlice,
            Y: trainingSetOutputSlice
        };

        helpers[k].send(setup);
    }

    var addToDerivates = function addToDerivates (r) {
        D1 = la.add2DMatrices(D1, r.D1);
        D2 = la.add2DMatrices(D2, r.D2);
        D3 = la.add2DMatrices(D3, r.D3);
    };

    var makeAStep = function () {
        thetas.Theta1 = la.sub2DMatrices(thetas.Theta1, la.mul2DMatrixByScalar(D1, learningRate));
        thetas.Theta2 = la.sub2DMatrices(thetas.Theta2, la.mul2DMatrixByScalar(D2, learningRate));
        thetas.Theta3 = la.sub2DMatrices(thetas.Theta3, la.mul2DMatrixByScalar(D3, learningRate));
    };

    var computeThetaScalar = function () {
        thetaVecScalar = la.sumOfSquares(thetas.Theta1);
        thetaVecScalar += la.sumOfSquares(thetas.Theta2);
        thetaVecScalar += la.sumOfSquares(thetas.Theta3);
        thetaVecScalar = Math.sqrt(thetaVecScalar);
    };

    var computeGradientScalar = function () {
        gradientScalar = la.sumOfSquares(D1);
        gradientScalar += la.sumOfSquares(D2);
        gradientScalar += la.sumOfSquares(D3);
        gradientScalar = Math.sqrt(gradientScalar);
    };

    var getNewSetup = function () {
        return {
            thetas : thetas,
            cost : totalCost,
            gradientSize : gradientScalar,
            thetaVecScalar : thetaVecScalar
        };
    };

    var cleanupHelpers = function () {
        for (k = 0; k < numberOfNodes; k++) {
            helpers[k].removeAllListeners();
        }
    };

    function processHelperResult (helperResult) {

        var r = helperResult;

        addToDerivates(r);

        totalCost += r.cost;

        allNodesFinished = --trainingRegressionCounter === 0;

        if (allNodesFinished) {

            computeThetaScalar();

            computeGradientScalar();

            makeAStep();

            cleanupHelpers();

            var setup = getNewSetup();

            var t = setTimeout(function () {
                callback(null, setup);
            }, 1);
        }
    };
};

module.exports = OptimisationStepExecutor;