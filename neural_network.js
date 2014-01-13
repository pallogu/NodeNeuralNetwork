var _ = require('underscore');
var numeric = require('numeric');
var os = require('os');
const computecluster = require('compute-cluster');
var cc = new computecluster({
    module: './modules/neuralNetworkWorker.js'
});
var model = [];


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

       var stepCounter = 0;


       var initialThetaVec = numeric.sub(numeric.random([1, (numberOfFeatures + 1) * numberOfActivationUnitsL1 + (numberOfActivationUnitsL1 + 1) * numberOfActivationUnitsL2 + numberOfActivationUnitsL2 + 1])[0], 0.5);


       var computeBatchStep = function() {
           var counter = numberOfNodes;
           var gradSum = numeric.rep([initialThetaVec.length],0);

           for (var k = 0; k < numberOfNodes; k++) {

               cc.enqueue({
                   numberOfFeatures: numberOfFeatures,
                   numberOfActivationUnitsL1: numberOfActivationUnitsL1,
                   numberOfActivationUnitsL2: numberOfActivationUnitsL2,
                   ThetaVec: initialThetaVec,
                   X: trainingSetX.slice(stepCounter + numberOfExamplesPerNode*k, stepCounter + numberOfExamplesPerNode*k + numberOfExamplesPerNode),
                   Y: trainingSetY.slice(stepCounter + numberOfExamplesPerNode*k, stepCounter + numberOfExamplesPerNode*k + numberOfExamplesPerNode)

               }, function (err, r) {
                   gradSum = numeric.add(gradSum, r[1]);
                   console.log('cost', r[0]);

                   if (--counter === 0) {
                       stepCounter = stepCounter + numberOfExamplesPerNode*numberOfCpus;
                       initialThetaVec = numeric.sub(initialThetaVec,  numeric.mul(gradientDescentAlpha/numberOfNodes, gradSum));

                       if(stepCounter < trainingSet.length - numberOfExamplesPerNode*numberOfNodes) {
                           computeBatchStep();
                       } else {
                           if(r[0] < maxCostError) {
                               console.log('finished');
                               model = initialThetaVec;
                           } else {

                               reshuffledTrainingSet = shuffler.reshuffle(numeric.clone(trainingSetX), numeric.clone(trainingSetY));

                               trainingSetX = reshuffledTrainingSet[0];
                               trainingSetY = reshuffledTrainingSet[1];
                               stepCounter = 0;

                               computeBatchStep();
                           }
                       }
                   }

               });

           }
       };

       computeBatchStep();
   },
   predict: function () {

   },
   computeNumericGradientDelta: function () {]

   }
});

module.exports = Neural_Network;