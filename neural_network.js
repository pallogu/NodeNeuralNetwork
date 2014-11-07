var _ = require('lodash');
var os = require('os');
var path = require('path');

var util = require('util');

var knuthShuffle = require('knuth-shuffle').knuthShuffle;
var La = require('./helpers/linear_algebra.helper.js');

var cp = require('child_process');

var memwatch = require('memwatch');

//memwatch.on('stats', function(stats) { console.log('\n\n', stats) });


//memwatch.on('leak', function(info) {
//    console.log(info);
//});

//var stdin = process.stdin;
//stdin.setRawMode( true );

// resume stdin in the parent process (node app won't quit all by itself
// unless an error or process.exit() happens)
//stdin.resume();
//
//stdin.setEncoding( 'utf8' );

//const ComputeCluster = require('compute-cluster');
//var computeCluster = new ComputeCluster({
//    module: path.join(__dirname, 'helpers/neural_network.helper.js')
//});

var Neural_Network = function () {};

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

_.extend(Neural_Network.prototype, {
    train: function (options, callback) {
        console.log('startTraining');
        var self = this;

//        var processPaused = false;
//        var stdinHandler = function( key ){
//            if ( key === '\u001B' ) {
//                console.log('press y to exit');
//                processPaused = true;
//            } else if ( processPaused && key === '\u0079') {
//                console.log('model', thetas.Theta1.vector.concat(thetas.Theta2.vector, thetas.Theta3.vector));
//                process.exit();
//            } else if (key === '\u0003') {
//                process.exit();
//            } else {
//                processPaused = false;
//            }
//        };
//
//        stdin.on( 'data', stdinHandler);
        var la = new La();


        var initialThetaVec = options.model || la.randomVector((options.trainingSetInput[0].length + 1) * options.numberOfActivationUnitsL1 + (options.numberOfActivationUnitsL1 + 1) * options.numberOfActivationUnitsL2 + options.numberOfActivationUnitsL2 + 1, 0.5);

        this.numberOfFeatures = options.trainingSetInput[0].length;
        this.numberOfOutputUnits = options.trainingSetOutput[0].length;
        this.numberOfActivationUnitsL1 = options.numberOfActivationUnitsL1;
        this.numberOfActivationUnitsL2 = options.numberOfActivationUnitsL2;

        var thetas = splitThetaIntoMatrices({
            ThetaVec: initialThetaVec,
            numberOfActivationUnitsL1: options.numberOfActivationUnitsL1,
            numberOfActivationUnitsL2: options.numberOfActivationUnitsL2,
            numberOfFeatures: options.trainingSetInput[0].length,
            numberOfOutputUnits: options.trainingSetOutput[0].length
        });

        this.numberOfNodes = options.numberOfNodes || os.cpus().length;
        this.numberOfExamplesPerNode = options.numberOfExamplesPerNode || 1;
        this.maxCostError =  options.maxCostError || 0.01;
        this.maxGradientSize = options.maxGradientSize || 1e-10;
        this.learningRate = options.learningRate || 1;
        this.maxNumberOfIterations = options.maxNoOfIterations || Number.MAX_VALUE;
        this.lambda = options.lambda;
        this.verboseMode = options.verboseMode || false;
        this.numberOfOptimizingIterations = 0;
        this.trainingSetInput = options.trainingSetInput;
        this.trainingSetOutput = options.trainingSetOutput;

        this.executeTrainingLoop(JSON.stringify(thetas), callback);

    },
    executeTrainingLoop: function (stringifiedThetas, callback) {
        global.gc();
        var hd = new memwatch.HeapDiff();

        var self = this;

        var la = new La();
        var thetas = JSON.parse(stringifiedThetas);
        stringifiedThetas = null;

        var D1 = la.create2DMatrix(thetas.Theta1.length, thetas.Theta1[0].length);
        var D2 = la.create2DMatrix(thetas.Theta2.length, thetas.Theta2[0].length);
        var D3 = la.create2DMatrix(thetas.Theta3.length, thetas.Theta3[0].length);

        var trainingSetSliceStart = 0;
        var trainingSetSliceEnd = 0;
        var trainingSetInputSlice = [] ;
        var trainingSetOutputSlice = [];
        var trainingRegressionCounter = self.numberOfNodes;

        var gradientScalar = 0;
        var thetaVecScalar = 0;
        var totalCost = 0;
        var batchSize = 0;
        var k;
        var allNodesFinished = false;


        var helpers = [];

        for (k = 0; k < self.numberOfNodes; k++) {
            var helper = cp.fork(path.join(__dirname, 'helpers/neural_network.helper.js'));
            helper.once('message', processHelperResult);
            helpers.push(helper);

            trainingSetSliceStart = self.numberOfExamplesPerNode * k;
            trainingSetSliceEnd = self.numberOfExamplesPerNode * (k + 1);
            trainingSetInputSlice = this.trainingSetInput.slice(trainingSetSliceStart, trainingSetSliceEnd);
            trainingSetOutputSlice = this.trainingSetOutput.slice(trainingSetSliceStart, trainingSetSliceEnd);

            batchSize += trainingSetInputSlice.length;

            var setup = JSON.stringify({
                batchSize: batchSize,
                Theta1 : thetas.Theta1,
                Theta2 : thetas.Theta2,
                Theta3 : thetas.Theta3,
                lambda: self.lambda,
                X: trainingSetInputSlice,
                Y: trainingSetOutputSlice
            });

            trainingSetInputSlice = null;
            trainingSetOutputSlice = null;

            helpers[k].send(setup);
            setup  = null;
        }


        var endTraining = function endTraining (totalCost, callback) {
            console.log('finished with final cost: ', totalCost);
            console.timeEnd('Time required to train:');

//            stdin.pause();
//            stdin.removeAllListeners();
//            stdin = null;
            for(k = 0; k < self.numberOfNodes; k++) {
                helpers[k].kill('SIGKILL');
            }

            callback.call(self, null, _.flatten(thetas.Theta1).concat(thetas.Theta2).concat(thetas.Theta3));
        };

        var reportProgress = function reportProgress () {
            if (self.verboseMode && self.numberOfOptimizingIterations % 1 === 0) {
//                global.gc();
                console.log('%d, %d', self.numberOfOptimizingIterations, util.inspect(process.memoryUsage().rss));
                console.log('Number of optimizing iterations: %s, current cost: %s, gradient scalar: %s, thetaVecSize: %s', self.numberOfOptimizingIterations, totalCost, gradientScalar, thetaVecScalar);
            }
        };

        var reshuffleTrainingSet = function reshuffleTrainingSet (trainingSetInput,trainingSetOutput) {

            var reshuffledTrainingSet = [];
            reshuffledTrainingSet = reshuffle(la.clone2dMatrix(trainingSetInput), la.clone2dMatrix(trainingSetOutput));

            trainingSetInput = reshuffledTrainingSet[0];
            trainingSetOutput = reshuffledTrainingSet[1];
        };

        var addToDerivates = function addToDerivates (r) {
            D1 = la.add2DMatrices(D1, r.D1);
            D2 = la.add2DMatrices(D2, r.D2);
            D3 = la.add2DMatrices(D3, r.D3);
        }

        var makeAStep = function () {
            thetas.Theta1 = la.sub2DMatrices(thetas.Theta1, la.mul2DMatrixByScalar(D1, self.learningRate));
            thetas.Theta2 = la.sub2DMatrices(thetas.Theta2, la.mul2DMatrixByScalar(D2, self.learningRate));
            thetas.Theta3 = la.sub2DMatrices(thetas.Theta3, la.mul2DMatrixByScalar(D3, self.learningRate));
        }

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
            return JSON.stringify(thetas);
        };

        function processHelperResult (helperResult) {

            var r = JSON.parse(helperResult)

            addToDerivates(r);
            totalCost += r.cost;

            r = null;
            helperResult = null;

            allNodesFinished = --trainingRegressionCounter === 0;

            if (allNodesFinished) {

                computeThetaScalar();

                computeGradientScalar();

                ++self.numberOfOptimizingIterations;
                reportProgress();

                if (self.numberOfOptimizingIterations > self.maxNoOfIterations || totalCost < self.maxCostError || gradientScalar < self.maxGradientSize) {
                    endTraining(totalCost, callback);
                } else {

                    reshuffleTrainingSet(self.trainingSetInput, self.trainingSetOutput);
                    makeAStep();

                    var setup = getNewSetup();

                    var t = setTimeout(function () {
                        for (k = 0; k < self.numberOfNodes; k++) {
                            helpers[k].removeAllListeners();
                            helpers[k].kill('SIGKILL');

                        }
                        la = null;
                        D1 = null;
                        D2 = null;
                        D3 = null;
                        thetas.Theta1 = null;
                        thetas.Theta2 = null;
                        thetas.Theta3 = null;
                        thetas = null;
                        trainingSetSliceStart = null;
                        trainingSetSliceEnd = null;
                        trainingSetInputSlice = null ;
                        trainingSetOutputSlice = null;
                        trainingRegressionCounter = null;
                        gradientScalar = null;
                        thetaVecScalar = null;
                        totalCost = null;
                        batchSize = null;
                        k = null;
                        allNodesFinished = null;
                        addToDerivates = null;
                        reshuffleTrainingSet = null;
                        reportProgress = null;
                        endTraining = null;


                        t = null;

                        self.executeTrainingLoop.call(self, setup, callback);
                        setup = null;
                        self = null;
                        processHelperResult = null;

                        global.gc();
                        var diff = hd.end();
                        console.log(diff.change);

                    },1);
                }
            }
        };

        console.time('Time required to train:');
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


        var setup = JSON.stringify({
            batchSize : 1,
            Theta1: thetas.Theta1,
            Theta2: thetas.Theta2,
            Theta3: thetas.Theta3,
            lambda: 0,
            X: [X],
            Y: [[0]]
        });

        var processHelperResult = function (helperResult) {
            helper.kill('SIGHUP');
            callback.call(that, null, JSON.parse(helperResult).prediction[0][0]);
        };

        var helper = cp.fork(path.join(__dirname, 'helpers/neural_network.helper.js'));
        helper.on('message', processHelperResult);

        helper.send(setup);
    }
});

module.exports = Neural_Network;