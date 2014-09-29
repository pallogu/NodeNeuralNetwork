var _ = require('underscore');
var numeric = require('numeric');
var M = require('eigenjs').Matrix;
var os = require('os');
var path = require('path');


const ComputeCluster = require('compute-cluster');
var computeCluster = new ComputeCluster({
    module: path.join(__dirname, 'helpers/neural_network.helper.js')
});


var shuffler = require('./helpers/shuffler.js');

var splitThetaVecToMatrices = function (setup) {

    var ThetaVec = setup.ThetaVec;
    var numberOfFeatures = setup.numberOfFeatures;
    var numberOfActivationUnitsL1 = setup.numberOfActivationUnitsL1;
    var numberOfActivationUnitsL2 = setup.numberOfActivationUnitsL2;
    var numberOfOutputUnits = setup.numberOfOutputUnits;

    var Theta1 = new M(numberOfFeatures + 1, numberOfActivationUnitsL1);
    var Theta2 = new M(numberOfActivationUnitsL1 + 1, numberOfActivationUnitsL2);
    var Theta3 = new M (numberOfActivationUnitsL2 + 1, numberOfOutputUnits);
    var theta1Vec = ThetaVec.slice(0, (numberOfFeatures + 1) * numberOfActivationUnitsL1);;
    var theta2Vec = ThetaVec.slice((numberOfFeatures + 1) * numberOfActivationUnitsL1, (numberOfActivationUnitsL1 + 1) * (numberOfActivationUnitsL2) + (numberOfFeatures + 1) * numberOfActivationUnitsL1);;
    var theta3Vec = ThetaVec.slice((numberOfActivationUnitsL1 + 1) * (numberOfActivationUnitsL2) + (numberOfFeatures + 1) * numberOfActivationUnitsL1, ThetaVec.length);;

    Theta1.set(theta1Vec);
    Theta2.set(theta2Vec);
    Theta3.set(theta3Vec);

    return {
        Theta1 : Theta1,
        Theta2 : Theta2,
        Theta3 : Theta3
    }
};

var Neural_Network = function () {
};

_.extend(Neural_Network.prototype, {
    exit: function (){
        computeCluster.exit();
    },
    train: function (options, callback) {
        var that = this;

        var trainingSetInput = options.trainingSetInput;
        var trainingSetOutput = options.trainingSetOutput;
        var numberOfNodes = options.numberOfNodes || os.cpus().length;
        var numberOfExamplesPerNode = options.numberOfExamplesPerNode || 1;
        var maxCostError = options.maxCostError || 0.01;
        var maxGradientSize = options.maxGradientSize || 1e-10;
        var learningRate = options.learningRate || 1;
        var maxNoOfIterations = options.maxNoOfIterations || Number.MAX_VALUE;
        var verboseMode = options.verboseMode || false;
        var numberOfProcessedExamples = 0;
        var numberOfOptimizingIterations = 0;
        var lambda = options.lambda || 1e-10;

        var numberOfFeatures = trainingSetInput[0].length;
        var numberOfOutputUnits = trainingSetOutput[0].length;
        var numberOfActivationUnitsL1 = options.numberOfActivationUnitsL1;
        var numberOfActivationUnitsL2 = options.numberOfActivationUnitsL2;

        var tmp = (numberOfFeatures + 1) * numberOfActivationUnitsL1 + (numberOfActivationUnitsL1 + 1) * numberOfActivationUnitsL2 + numberOfActivationUnitsL2 + numberOfOutputUnits;

        var initialThetaVec = options.model || M.Random(1, tmp);

        var thetaMatrices = splitThetaVecToMatrices({
            ThetaVec: initialThetaVec,
            numberOfActivationUnitsL1: numberOfActivationUnitsL1,
            numberOfActivationUnitsL2: numberOfActivationUnitsL2,
            numberOfFeatures: numberOfFeatures,
            numberOfOutputUnits: numberOfOutputUnits
        });


        var endTraining = function (err, totalCost, initialThetaVec, callback) {
            console.log('finished with final cost: ', totalCost);
            console.timeEnd('Time required to train:');

            callback.call(that, err, initialThetaVec);
        };
        var reportProgress = function (totalCost, gradientScalar, thetaVecScalar) {
            if (verboseMode && numberOfOptimizingIterations % 1000 === 0) {
                console.log('Number of optimizing iterations: %s, current cost: %s, gradient scalar: %s, thetaVecSize: %s', numberOfOptimizingIterations, totalCost, gradientScalar, thetaVecScalar);
            }
        };
        var reshuffle = function (trainingSetInput,trainingSetOutput) {
            reshuffledTrainingSet = shuffler.reshuffle(numeric.clone(trainingSetInput), numeric.clone(trainingSetOutput));

            trainingSetInput = reshuffledTrainingSet[0];
            trainingSetOutput = reshuffledTrainingSet[1];
        };
        var resetNumberOfProcessedExamples = function () {numberOfProcessedExamples = 0;};
        var stepInGradientDirection = function (sumOfGradientsFromNodes) { initialThetaVec = numeric.sub(initialThetaVec, numeric.mul(learningRate / numberOfNodes, sumOfGradientsFromNodes));}

        var processTrainingExamples = function () {
            var trainingRegressionCounter = numberOfNodes;
            var sumOfGradientsFromNodes = numeric.rep([initialThetaVec.length], 0);
            var gradientScalar = 0;
            var thetaVecScalar = 0;
            var totalCost = 0;
            var batchSize = 0;

            for (var k = 0; k < numberOfNodes; k++) {
                var trainingSetSliceStart = numberOfProcessedExamples + numberOfExamplesPerNode * k;
                var trainingSetSliceEnd = numberOfProcessedExamples + numberOfExamplesPerNode * (k + 1);
                var trainingSetInputSlice = trainingSetInput.slice(trainingSetSliceStart, trainingSetSliceEnd);
                var trainingSetOutputSlice = trainingSetOutput.slice(trainingSetSliceStart, trainingSetSliceEnd);

                batchSize += trainingSetInputSlice.length;

                computeCluster.enqueue({
                    Theta1 : thetaMatrices.Theta1,
                    Theta2 : thetaMatrices.Theta2,
                    Theta3 : thetaMatrices.Theta3,
                    lambda: lambda,
                    X: trainingSetInputSlice,
                    Y: trainingSetOutputSlice

                }, function (err, nnTrainingCoreResult) {
                    var gradientFromCurrentBatch = nnTrainingCoreResult[1];
                    var costFromCurrentBatch = nnTrainingCoreResult[0]
                    sumOfGradientsFromNodes = numeric.add(sumOfGradientsFromNodes, gradientFromCurrentBatch);
                    totalCost += costFromCurrentBatch;
                    numberOfProcessedExamples += numberOfExamplesPerNode;

                    var allNodesFinished = --trainingRegressionCounter === 0;

                    if (allNodesFinished) {
                        sumOfGradientsFromNodes = numeric.mul(1/batchSize, sumOfGradientsFromNodes);
                        totalCost = totalCost / batchSize;

                        thetaVecScalar = Math.sqrt(_.reduce(initialThetaVec, function (sum, num) { return (sum + num * num);}, 0));
                        gradientScalar = Math.sqrt(_.reduce(sumOfGradientsFromNodes, function (sum, num) { return (sum + num * num);}, 0))/initialThetaVec.length;

                        if (numberOfProcessedExamples < trainingSetInput.length - numberOfExamplesPerNode * numberOfNodes) {
                            stepInGradientDirection(sumOfGradientsFromNodes);
                            processTrainingExamples();
                        } else {
                            ++numberOfOptimizingIterations;
                            reportProgress(totalCost, gradientScalar, thetaVecScalar);

                            if (numberOfOptimizingIterations > maxNoOfIterations || totalCost < maxCostError || gradientScalar < maxGradientSize) {
                                endTraining(err, totalCost, initialThetaVec, callback);
                            } else {
                                resetNumberOfProcessedExamples();
                                reshuffle(trainingSetInput, trainingSetOutput);
                                stepInGradientDirection(sumOfGradientsFromNodes);
                                processTrainingExamples();
                            }
                        }
                    }
                });
            }
        };
        console.time('Time required to train:');
        processTrainingExamples();
    },
    predict: function (opts, callback) {
        var that = this;
        var X = opts.inputVector;
        var numberOfActivationUnitsL1 = opts.numberOfActivationUnitsL1;
        var numberOfActivationUnitsL2 = opts.numberOfActivationUnitsL2;
        var model = opts.model;

        computeCluster.enqueue({
            numberOfFeatures: X.length,
            numberOfActivationUnitsL1: numberOfActivationUnitsL1,
            numberOfActivationUnitsL2: numberOfActivationUnitsL2,
            ThetaVec: model,
            X: [X],
            Y: [[0]]

        }, function (err, nnTrainingCoreResult) {
            callback.call(that, err, nnTrainingCoreResult[2][0]);
        });
    }
});

module.exports = Neural_Network;
