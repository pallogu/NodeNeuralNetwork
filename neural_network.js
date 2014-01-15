var _ = require('underscore');
var numeric = require('numeric');
var os = require('os');
var path = require('path');


const ComputeCluster = require('compute-cluster');
var computeCluster = new ComputeCluster({
    module: path.join(__dirname, 'helpers/neural_network.helper.js')
});


var shuffler = require('./helpers/shuffler.js');

var model = [];

var numberOfFeatures;
var numberOfActivationUnitsL1;
var numberOfActivationUnitsL2;

var Neural_Network = function () {
};

_.extend(Neural_Network.prototype, {
    exit: function (){
        computeCluster.exit();
    },
    train: function (options, callback) {
        var that = this;

        var trainingSetX = options.trainingSetX;
        var trainingSetY = options.trainingSetY;
        var numberOfNodes = options.numberOfNodes || os.cpus.length;
        var numberOfExamplesPerNode = options.numberOfExamplesPerNode || 1;
        var maxCostError = options.maxCostError || 0.01;
        var learningRate = options.learningRate || 1;
        var maxNoOfIterations = options.maxNoOfIterations || Number.MAX_VALUE;
        var numberOfProcessedExamples = 0;
        var numberOfOptimizingIterations = 0;
        var initialThetaVec = [];

        numberOfFeatures = trainingSetX[0].length
        numberOfActivationUnitsL1 = options.numberOfActivationUnitsL1;
        numberOfActivationUnitsL2 = options.numberOfActivationUnitsL2;
        initialThetaVec = numeric.sub(numeric.random([1, (numberOfFeatures + 1) * numberOfActivationUnitsL1 + (numberOfActivationUnitsL1 + 1) * numberOfActivationUnitsL2 + numberOfActivationUnitsL2 + 1])[0], 0.5);

        console.time('Time required to train:');

        var processTrainingExamples = function () {
            var trainingRegressionCounter = numberOfNodes;
            var sumOfGradientsFromNodes = numeric.rep([initialThetaVec.length], 0);

            for (var k = 0; k < numberOfNodes; k++) {

                computeCluster.enqueue({
                    numberOfFeatures: numberOfFeatures,
                    numberOfActivationUnitsL1: numberOfActivationUnitsL1,
                    numberOfActivationUnitsL2: numberOfActivationUnitsL2,
                    ThetaVec: initialThetaVec,
                    X: trainingSetX.slice(numberOfProcessedExamples + numberOfExamplesPerNode * k, numberOfProcessedExamples + numberOfExamplesPerNode * k + numberOfExamplesPerNode),
                    Y: trainingSetY.slice(numberOfProcessedExamples + numberOfExamplesPerNode * k, numberOfProcessedExamples + numberOfExamplesPerNode * k + numberOfExamplesPerNode)

                }, function (err, nnTrainingCoreResult) {

                    sumOfGradientsFromNodes = numeric.add(sumOfGradientsFromNodes, nnTrainingCoreResult[1]);

                    //console.log('cost', nnTrainingCoreResult[0]);

                    if (--trainingRegressionCounter === 0) {

                        numberOfProcessedExamples = numberOfProcessedExamples + numberOfExamplesPerNode * numberOfNodes;

                        initialThetaVec = numeric.sub(initialThetaVec, numeric.mul(learningRate / numberOfNodes, sumOfGradientsFromNodes));

                        if (numberOfProcessedExamples < trainingSetX.length - numberOfExamplesPerNode * numberOfNodes) {

                            processTrainingExamples();

                        } else {
                            ++numberOfOptimizingIterations;
                            //console.log('Number of optimizing iterations: %s, current cost: %s', numberOfOptimizingIterations, nnTrainingCoreResult[0]);

                            if (nnTrainingCoreResult[0] < maxCostError || numberOfOptimizingIterations > maxNoOfIterations) {

                                console.log('finished with final cost: ', nnTrainingCoreResult[0]);
                                console.timeEnd('Time required to train:');

                                model = initialThetaVec;

                                callback.call(that, err, model);

                            } else {

                                reshuffledTrainingSet = shuffler.reshuffle(numeric.clone(trainingSetX), numeric.clone(trainingSetY));

                                trainingSetX = reshuffledTrainingSet[0];
                                trainingSetY = reshuffledTrainingSet[1];

                                numberOfProcessedExamples = 0;

                                processTrainingExamples();
                            }
                        }
                    }
                });
            }
        };

        processTrainingExamples();
    },
    predict: function (X, callback) {
        var that = this;
        computeCluster.enqueue({
            numberOfFeatures: numberOfFeatures,
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
