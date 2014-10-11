var _ = require('lodash');
var os = require('os');
var path = require('path');

var knuthShuffle = require('knuth-shuffle').knuthShuffle;
var la = require('./helpers/linear_algebra.helper.js');

var stdin = process.stdin;
stdin.setRawMode( true );

// resume stdin in the parent process (node app won't quit all by itself
// unless an error or process.exit() happens)
stdin.resume();

stdin.setEncoding( 'utf8' );

const ComputeCluster = require('compute-cluster');
var computeCluster = new ComputeCluster({
    module: path.join(__dirname, 'helpers/neural_network.helper.js')
});

var Neural_Network = function () {};

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

var tmpArray = [];
var tmpX = [];
var tmpY = [];
var shufflerCounter= 0;

var reshuffle = function (Xmatrix, Ymatrix) {
    'use strict';

    tmpArray = [];
    tmpX = [];
    tmpY = [];
    shufflerCounter;

    if(Xmatrix.length !== Ymatrix.length) {
        throw 'Shuffler: reshuffle method: Length of arrays do not match';
    } else {
        for(shufflerCounter = 0; shufflerCounter < Xmatrix.length; shufflerCounter = shufflerCounter + 1) {
            tmpArray.push([Xmatrix[shufflerCounter], Ymatrix[shufflerCounter]]);
        }

        knuthShuffle(tmpArray);

        for(shufflerCounter = 0; shufflerCounter < Xmatrix.length; shufflerCounter = shufflerCounter + 1) {
            tmpX.push(tmpArray[shufflerCounter][0]);
            tmpY.push(tmpArray[shufflerCounter][1]);
        }
    }
    tmpArray = [];
    shufflerCounter = 0;

    return [tmpX, tmpY];
};

_.extend(Neural_Network.prototype, {
    exit: function (){
        computeCluster.exit();
    },
    train: function (options, callback) {
        console.log('starting training');
        var that = this;

        var processPaused = false;
        var stdinHandler = function( key ){
            if ( key === '\u001B' ) {
                console.log('press y to exit');
                processPaused = true;
            } else if ( processPaused && key === '\u0079') {
                console.log('model', thetas.Theta1.vector.concat(thetas.Theta2.vector, thetas.Theta3.vector));
                process.exit();
            } else if (key === '\u0003') {
                process.exit();
            } else {
                processPaused = false;
            }
        };

        stdin.on( 'data', stdinHandler);

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
        var reshuffledTrainingSet = [];

        var numberOfFeatures = trainingSetInput[0].length;
        var numberOfOutputUnits = trainingSetOutput[0].length;
        var numberOfActivationUnitsL1 = options.numberOfActivationUnitsL1;
        var numberOfActivationUnitsL2 = options.numberOfActivationUnitsL2;

        var initialThetaVec = options.model || la.randomVector((numberOfFeatures + 1) * numberOfActivationUnitsL1 + (numberOfActivationUnitsL1 + 1) * numberOfActivationUnitsL2 + numberOfActivationUnitsL2 + 1, 0.5);

        var thetas = splitThetaIntoMatrices({
            ThetaVec: initialThetaVec,
            numberOfActivationUnitsL1: numberOfActivationUnitsL1,
            numberOfActivationUnitsL2: numberOfActivationUnitsL2,
            numberOfFeatures: numberOfFeatures,
            numberOfOutputUnits: numberOfOutputUnits
        });


        var endTraining = function (err, callback) {
            console.log('finished with final cost: ', totalCost);
            console.timeEnd('Time required to train:');

            stdin.pause();
            stdin.removeAllListeners();
            stdin = null;

            callback.call(that, err, _.flatten(thetas.Theta1).concat(_.flatten(thetas.Theta2), _.flatten(thetas.Theta3)));
        };
        var reportProgress = function (totalCost, gradientScalar, thetaVecScalar) {
            if (verboseMode && numberOfOptimizingIterations % 1000 === 0) {
                console.log('Number of optimizing iterations: %s, current cost: %s, gradient scalar: %s, thetaVecSize: %s', numberOfOptimizingIterations, totalCost, gradientScalar, thetaVecScalar);
            }
        };
        var reshuffleTrainingSet = function (trainingSetInput,trainingSetOutput) {
            reshuffledTrainingSet = reshuffle(la.clone2dMatrix(trainingSetInput), la.clone2dMatrix(trainingSetOutput));

            trainingSetInput = reshuffledTrainingSet[0];
            trainingSetOutput = reshuffledTrainingSet[1];
        };
        var resetNumberOfProcessedExamples = function () {numberOfProcessedExamples = 0;};
        var stepInGradientDirection = function (D1, D2, D3) {
            thetas.Theta1 = la.sub2DMatrices(thetas.Theta1, la.mul2DMatrixByScalar(D1, learningRate, D1));
            thetas.Theta2 = la.sub2DMatrices(thetas.Theta2, la.mul2DMatrixByScalar(D2, learningRate, D2));
            thetas.Theta3 = la.sub2DMatrices(thetas.Theta3, la.mul2DMatrixByScalar(D3, learningRate, D3));
        };

        var trainingRegressionCounter = 0;
        var D1 = [];
        var D2 = [];
        var D3 = [];

        var gradientScalar = 0;
        var thetaVecScalar = 0;
        var totalCost = 0;
        var batchSize = 0;

        var trainingSetSliceStart = 0;
        var trainingSetSliceEnd = 0;
        var trainingSetInputSlice = [] ;
        var trainingSetOutputSlice = [];
        var k;
        var allNodesFinished = false;


        var processTrainingExamples = function () {

            trainingRegressionCounter = numberOfNodes;
            D1 = la.create2DMatrix(thetas.Theta1.length, thetas.Theta1[0].length);
            D2 = la.create2DMatrix(thetas.Theta2.length, thetas.Theta2[0].length);
            D3 = la.create2DMatrix(thetas.Theta3.length, thetas.Theta3[0].length);


            gradientScalar = 0;
            thetaVecScalar = 0;
            totalCost = 0;
            batchSize = 0;


            for (k = 0; k < numberOfNodes; k++) {
//                console.log('starting in node: ', k);
                trainingSetSliceStart = numberOfProcessedExamples + numberOfExamplesPerNode * k;
                trainingSetSliceEnd = numberOfProcessedExamples + numberOfExamplesPerNode * (k + 1);
                trainingSetInputSlice = trainingSetInput.slice(trainingSetSliceStart, trainingSetSliceEnd);
                trainingSetOutputSlice = trainingSetOutput.slice(trainingSetSliceStart, trainingSetSliceEnd);

                batchSize += trainingSetInputSlice.length;

                computeCluster.enqueue({
                    Theta1: thetas.Theta1,
                    Theta2: thetas.Theta2,
                    Theta3: thetas.Theta3,
                    lambda: lambda,
                    X: trainingSetInputSlice,
                    Y: trainingSetOutputSlice
                }, function (err, nnTrainingCoreResult) {

                    nnTrainingCoreResult = nnTrainingCoreResult;


                    D1 = la.add2DMatrices(D1, nnTrainingCoreResult.D1);
                    D2 = la.add2DMatrices(D2, nnTrainingCoreResult.D2);
                    D3 = la.add2DMatrices(D3, nnTrainingCoreResult.D3);
                    totalCost += nnTrainingCoreResult.cost;


                    allNodesFinished = --trainingRegressionCounter === 0;

                    if (allNodesFinished) {

                        numberOfProcessedExamples += batchSize;

                        D1 = la.mul2DMatrixByScalar(D1, 1/batchSize);
                        D2 = la.mul2DMatrixByScalar(D2, 1/batchSize);
                        D3 = la.mul2DMatrixByScalar(D3, 1/batchSize);

                        totalCost = totalCost / batchSize;

                        thetaVecScalar = _.reduce(_.flatten(thetas.Theta1), function (sum, num) { return (sum + num * num);}, 0);
                        thetaVecScalar += _.reduce(_.flatten(thetas.Theta2), function (sum, num) { return (sum + num * num);}, 0);
                        thetaVecScalar += _.reduce(_.flatten(thetas.Theta3), function (sum, num) { return (sum + num * num);}, 0);
                        thetaVecScalar = Math.sqrt(thetaVecScalar);

                        gradientScalar = _.reduce(_.flatten(D1), function (sum, num) { return (sum + num * num);}, 0);
                        gradientScalar += _.reduce(_.flatten(D2), function (sum, num) { return (sum + num * num);}, 0);
                        gradientScalar += _.reduce(_.flatten(D3), function (sum, num) { return (sum + num * num);}, 0);
                        gradientScalar = Math.sqrt(gradientScalar);

                        if (numberOfProcessedExamples < trainingSetInput.length - numberOfExamplesPerNode * numberOfNodes) {
                            stepInGradientDirection(D1, D2, D3);

                            trainingRegressionCounter = 0;
                            D1 = la.add2DMatrices(D1, nnTrainingCoreResult.D1);
                            D2 = la.add2DMatrices(D2, nnTrainingCoreResult.D2);
                            D3 = la.add2DMatrices(D3, nnTrainingCoreResult.D3);

                            gradientScalar = 0;
                            thetaVecScalar = 0;
                            totalCost = 0;
                            batchSize = 0;

                            processTrainingExamples();
                        } else {
                            ++numberOfOptimizingIterations;
                            reportProgress(totalCost, gradientScalar, thetaVecScalar);

                            if (numberOfOptimizingIterations > maxNoOfIterations || totalCost < maxCostError || gradientScalar < maxGradientSize) {
                                endTraining(err, callback);
                            } else {
                                resetNumberOfProcessedExamples();
                                reshuffleTrainingSet(trainingSetInput, trainingSetOutput);
                                stepInGradientDirection(D1, D2, D3);

                                trainingRegressionCounter = 0;
                                D1 = la.add2DMatrices(D1, nnTrainingCoreResult.D1);
                                D2 = la.add2DMatrices(D2, nnTrainingCoreResult.D2);
                                D3 = la.add2DMatrices(D3, nnTrainingCoreResult.D3);

                                gradientScalar = 0;
                                thetaVecScalar = 0;
                                totalCost = 0;
                                batchSize = 0;

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

        var thetas = splitThetaIntoMatrices({
            ThetaVec: model,
            numberOfActivationUnitsL1: numberOfActivationUnitsL1,
            numberOfActivationUnitsL2: numberOfActivationUnitsL2,
            numberOfFeatures: X.length,
            numberOfOutputUnits: 1
        });

        var setup = {
            Theta1: thetas.Theta1,
            Theta2: thetas.Theta2,
            Theta3: thetas.Theta3,
            lambda: 0,
            X: [X],
            Y: [[0]]
        };

        computeCluster.enqueue(setup, function (err, nnTrainingCoreResult) {
            callback.call(that, err, nnTrainingCoreResult.prediction[0][0]);
        });
    }
});

module.exports = Neural_Network;