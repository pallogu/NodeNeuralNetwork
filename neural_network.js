var _ = require('underscore');
var numeric = require('numeric');
var os = require('os');
var path = require('path');


const ComputeCluster = require('compute-cluster');
var computeCluster = new ComputeCluster({
    module: path.join(__dirname, 'helpers/neural_network.helper.js')
});


var shuffler = require('./helpers/shuffler.js');

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
        var maxGradientSize = options.maxGradientSize || 0.00001;
        var learningRate = options.learningRate || 1;
        var maxNoOfIterations = options.maxNoOfIterations || Number.MAX_VALUE;
        var verboseMode = options.verboseMode || false;
        var numberOfProcessedExamples = 0;
        var numberOfOptimizingIterations = 0;

        var numberOfFeatures = trainingSetInput[0].length
        var numberOfActivationUnitsL1 = options.numberOfActivationUnitsL1;
        var numberOfActivationUnitsL2 = options.numberOfActivationUnitsL2;

        var initialThetaVec = options.model || numeric.sub(numeric.random([1, (numberOfFeatures + 1) * numberOfActivationUnitsL1 + (numberOfActivationUnitsL1 + 1) * numberOfActivationUnitsL2 + numberOfActivationUnitsL2 + 1])[0], 0.5);
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

        var verifyMinimum = function (input, output, gradientAtTheta, callback) {
            var epsilon = 0.001;
            var completedPartialDerivations = 0;
            var diagonalHessianElementsPositive = true;

            for (var k = 0, l = initialThetaVec.length; k < l; k ++) {
                var epsilonTheta = numeric.clone(initialThetaVec);
                epsilonTheta[k] += epsilon;
                (function (index) {
                    computeCluster.enqueue({
                        numberOfFeatures: numberOfFeatures,
                        numberOfActivationUnitsL1: numberOfActivationUnitsL1,
                        numberOfActivationUnitsL2: numberOfActivationUnitsL2,
                        ThetaVec: epsilonTheta,
                        X: input,
                        Y: output

                    }, function (err, nnTrainingCoreResult) {
                        completedPartialDerivations++;
                        var secondaryDerivation = (nnTrainingCoreResult[1][index] - gradientAtTheta[index])/epsilon;
                        if(secondaryDerivation < 0) {
                            diagonalHessianElementsPositive = false;
                        }
                        if(completedPartialDerivations === (initialThetaVec.length - 1)){
                            callback(null, diagonalHessianElementsPositive);
                        }
                    });
                })(k);


            }
        };

        var processTrainingExamples = function () {
            var trainingRegressionCounter = numberOfNodes;
            var sumOfGradientsFromNodes = numeric.rep([initialThetaVec.length], 0);
            var gradientScalar = 0;
            var thetaVecScalar = 0;
            var totalCost = 0;

            for (var k = 0; k < numberOfNodes; k++) {
                var trainingSetSliceStart = numberOfProcessedExamples + numberOfExamplesPerNode * k;
                var trainingSetSliceEnd = numberOfProcessedExamples + numberOfExamplesPerNode * (k + 1);
                var trainingSetInputSlice = trainingSetInput.slice(trainingSetSliceStart, trainingSetSliceEnd);
                var trainingSetOutputSlice = trainingSetOutput.slice(trainingSetSliceStart, trainingSetSliceEnd);

                computeCluster.enqueue({
                    numberOfFeatures: numberOfFeatures,
                    numberOfActivationUnitsL1: numberOfActivationUnitsL1,
                    numberOfActivationUnitsL2: numberOfActivationUnitsL2,
                    ThetaVec: initialThetaVec,
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
                        thetaVecScalar = Math.sqrt(_.reduce(initialThetaVec, function (sum, num) { return (sum + num * num);}, 0));
                        gradientScalar = Math.sqrt(_.reduce(sumOfGradientsFromNodes, function (sum, num) { return (sum + num * num);}, 0))/initialThetaVec.length;

                        if (numberOfProcessedExamples < trainingSetInput.length - numberOfExamplesPerNode * numberOfNodes) {
                            stepInGradientDirection(sumOfGradientsFromNodes);
                            processTrainingExamples();
                        } else {
                            ++numberOfOptimizingIterations;
                            reportProgress(totalCost, gradientScalar, thetaVecScalar);

                            if (numberOfOptimizingIterations > maxNoOfIterations || totalCost < maxCostError) {
                                endTraining(err, totalCost, initialThetaVec, callback);
                            } else {
                                if(gradientScalar < maxGradientSize) {
                                    var batchInputSlice = trainingSetInput.slice(numberOfProcessedExamples - numberOfNodes*numberOfExamplesPerNode, numberOfProcessedExamples);
                                    var batchOutputSlice = trainingSetOutput.slice(numberOfProcessedExamples - numberOfNodes*numberOfExamplesPerNode, numberOfProcessedExamples);

                                    verifyMinimum(batchInputSlice, batchOutputSlice, sumOfGradientsFromNodes, function (err, descentIsAtMinimum) {
                                        if(descentIsAtMinimum) {
                                            endTraining(err, totalCost, initialThetaVec, callback)
                                        } else {
                                            resetNumberOfProcessedExamples();
                                            reshuffle(trainingSetInput, trainingSetOutput);
                                            stepInGradientDirection(sumOfGradientsFromNodes);
                                            processTrainingExamples();
                                        }
                                    });
                                } else {
                                    resetNumberOfProcessedExamples();
                                    reshuffle(trainingSetInput, trainingSetOutput);
                                    stepInGradientDirection(sumOfGradientsFromNodes);
                                    processTrainingExamples();
                                }
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
