var _ = require('underscore');
var numeric = require('numeric');
var os = require('os');


const ComputeCluster = require('compute-cluster');
var computeCluster = new ComputeCluster({
    module: './helpers/neuralNetworkWorker.js'
});


var model = [];

var shuffler = require('./helpers/shuffler.js');


var Neural_Network = function () {};

_.extend(Neural_Network.prototype, {
   train: function (options) {

       var trainingSetX = options.trainingSetX;
       var trainingSetY = options.trainingSetY;
       var numberOfNodes = options.numberOfNodes || os.cpus.length;
       var numberOfExamplesPerNode = options.numberOfExamplesPerNode || 1;
       var numberOfFeatures = trainingSetX[0][0].length
       var numberOfActivationUnitsL1 = options.numberOfActivationUnitsL1;
       var numberOfActivationUnitsL2 = options.numberOfActivationUnitsL2;
       var maxCostError = options.maxCost || 0.01;
       var gradientDescentAlpha = options.gradientDescentAlpha || 1;

       var numberOfProcessedExamples = 0;
       var initialThetaVec = numeric.sub(numeric.random([1, (numberOfFeatures + 1) * numberOfActivationUnitsL1 + (numberOfActivationUnitsL1 + 1) * numberOfActivationUnitsL2 + numberOfActivationUnitsL2 + 1])[0], 0.5);


       var processTrainingExamples = function() {
           var trainingRegressionCounter = numberOfNodes;
           var sumOfGradientsFromNodes = numeric.rep([initialThetaVec.length],0);

           for (var k = 0; k < numberOfNodes; k++) {

               computeCluster.enqueue({
                   numberOfFeatures: numberOfFeatures,
                   numberOfActivationUnitsL1: numberOfActivationUnitsL1,
                   numberOfActivationUnitsL2: numberOfActivationUnitsL2,
                   ThetaVec: initialThetaVec,
                   X: trainingSetX.slice(numberOfProcessedExamples + numberOfExamplesPerNode*k, numberOfProcessedExamples + numberOfExamplesPerNode*k + numberOfExamplesPerNode),
                   Y: trainingSetY.slice(numberOfProcessedExamples + numberOfExamplesPerNode*k, numberOfProcessedExamples + numberOfExamplesPerNode*k + numberOfExamplesPerNode)

               }, function (err, nnTrainingCoreResult) {

                   sumOfGradientsFromNodes = numeric.add(sumOfGradientsFromNodes, nnTrainingCoreResult[1]);

                   console.log('cost', nnTrainingCoreResult[0]);

                   if (--trainingRegressionCounter === 0) {

                       numberOfProcessedExamples = numberOfProcessedExamples + numberOfExamplesPerNode*numberOfNodes;

                       initialThetaVec = numeric.sub(initialThetaVec,  numeric.mul(gradientDescentAlpha/numberOfNodes, sumOfGradientsFromNodes));

                       if(numberOfProcessedExamples < trainingSetX.length - numberOfExamplesPerNode*numberOfNodes) {

                           processTrainingExamples();

                       } else {
                           if(nnTrainingCoreResult[0] < maxCostError) {

                               console.log('finished');
                               model = initialThetaVec;

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
   predict: function () {

   }
});

module.exports = Neural_Network;