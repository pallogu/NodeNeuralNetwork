var _ = require('lodash');
var os = require('os');
var cp = require('child_process');
var path = require('path');

var knuthShuffle = require('knuth-shuffle').knuthShuffle;
var La = require('./helpers/linear_algebra.helper.js');

var OptimisationStepExecutor = require('./helpers/optimisation_step_executor.helper.js');

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

    var theta1Vec = ThetaVec.slice(0, (numberOfFeatures + 1) * numberOfActivationUnitsL1);
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
        var la = La();


        var initialThetaVec = options.model || la.randomVector((options.trainingSetInput[0].length + 1) * options.numberOfActivationUnitsL1 + (options.numberOfActivationUnitsL1 + 1) * options.numberOfActivationUnitsL2 + options.numberOfActivationUnitsL2 + 1, 0.5);

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
        this.maxNoOfIterations = options.maxNoOfIterations || Number.MAX_VALUE;
        this.lambda = options.lambda;
        this.verboseMode = options.verboseMode || false;
        this.numberOfOptimizingIterations = 0;
        this.trainingSetInput = options.trainingSetInput;
        this.trainingSetOutput = options.trainingSetOutput;
        this.helpers = [];
        this.callback = callback;
        this.momentumCoefficient = options.momentumCoefficient || 0.9;

        for(var k = 0; k < this.numberOfNodes; k++) {
            this.helpers.push(cp.fork(path.join(__dirname, 'helpers/neural_network.helper.js')));
        }

        this.executeTrainingLoop(thetas);

        console.time('Time required to train:');
    },
    executeTrainingLoop: function (thetas) {

        var self = this;
        var la = La();

        reshuffleTrainingSet(self.trainingSetInput, self.trainingSetOutput);

        var setup = {
            thetas: thetas,
            pastThetas: self.pastThetas,
            learningRate : this.learningRate,
            lambda: this.lambda,
            numberOfExamplesPerNode :  this.numberOfExamplesPerNode,
            trainingSetInput : this.trainingSetInput,
            trainingSetOutput : this.trainingSetOutput,
            helpers : this.helpers,
            momentumCoefficient : this.momentumCoefficient
        };

        var optimisationStepExecutor = new OptimisationStepExecutor();

        var TMPTheta1 = la.clone2dMatrix(thetas.Theta1);
        var TMPTheta2 = la.clone2dMatrix(thetas.Theta2);
        var TMPTheta3 = la.clone2dMatrix(thetas.Theta3);

        optimisationStepExecutor.execute(setup, function (err, stepResult) {

            var parsedStepResult = stepResult;
            var cost = parsedStepResult.cost;
            var gradientSize = parsedStepResult.gradientSize;
            var thetaVecScalar = parsedStepResult.thetaVecScalar;
            var thetasAfterOptimisationStep = parsedStepResult.thetas;

            ++self.numberOfOptimizingIterations;
            reportProgress(cost, gradientSize, thetaVecScalar);


            if (self.numberOfOptimizingIterations > self.maxNoOfIterations || cost < self.maxCostError || gradientSize < self.maxGradientSize) {
                endTraining(thetasAfterOptimisationStep, cost);
            } else {
                self.pastThetas = {};
                self.pastThetas.Theta1 = TMPTheta1;
                self.pastThetas.Theta2 = TMPTheta2;
                self.pastThetas.Theta3 = TMPTheta3;
                optimisationStepExecutor = null;
                la = null;
                setup = null;
                self.executeTrainingLoop(thetasAfterOptimisationStep);
            }
        });

        function endTraining (thetasAfterOptimisationStep, totalCost) {
            console.log('finished with final cost: ', totalCost);
            console.timeEnd('Time required to train:');
            for(var i = 0; i < self.helpers.length; i++) {
                self.helpers[i].kill('SIGKILL');
                self.helpers[i] = null;
            }

            self.callback.call(self, null, _.flatten(thetasAfterOptimisationStep.Theta1).concat(_.flatten(thetasAfterOptimisationStep.Theta2)).concat(_.flatten(thetasAfterOptimisationStep.Theta3)));
        }

        function reportProgress (cost, gradientSize, thetaVecScalar) {
            if (self.verboseMode && self.numberOfOptimizingIterations % 1000 === 0) {
                console.log('Number of optimizing iterations: %s, current cost: %s, gradient scalar: %s, thetaVecSize: %s', self.numberOfOptimizingIterations, cost, gradientSize, thetaVecScalar);
            }
        }

        function reshuffleTrainingSet (trainingSetInput,trainingSetOutput) {

            var reshuffledTrainingSet = reshuffle(la.clone2dMatrix(trainingSetInput), la.clone2dMatrix(trainingSetOutput));

            trainingSetInput = reshuffledTrainingSet[0];
            trainingSetOutput = reshuffledTrainingSet[1];
        }
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
            batchSize : 1,
            Theta1: thetas.Theta1,
            Theta2: thetas.Theta2,
            Theta3: thetas.Theta3,
            lambda: 0,
            X: [X],
            Y: [[0]]
        };

        var processHelperResult = function (helperResult) {
            helper.kill('SIGHUP');
            callback.call(that, null, helperResult.prediction[0][0]);
        };

        var helper = cp.fork(path.join(__dirname, 'helpers/neural_network.helper.js'));
        helper.on('message', processHelperResult);

        helper.send(setup);
    }
});

module.exports = Neural_Network;