/*global process:true*/
var numeric = require('numeric');
var _ = require('underscore');

var numberOfFeatures;
var numberOfActivationUnitsL1;
var numberOfActivationUnitsL2;
var X;
var y;
var ThetaVec;
var numberOfExamples;

var convertMatrixToVector = function (matrix) {
    'use strict';

    return _.flatten(matrix);
};

var convertVectorToMatrix = function (vector, noRows, noCulumns) {
    'use strict';

    var matrix = [];

    if (vector.length !== noRows * noCulumns) {
        throw('can not convert vector to matrix because of invalid size');
    } else {
        for (var i = 0; i < noRows; i++) {
            matrix.push(vector.slice(i * noCulumns, (i + 1) * noCulumns));
        }
    }

    return matrix;
};

var addBias = function (activationUnits) {
    var activationUnitsWithBias = numeric.clone(activationUnits);
    for (i = 0; i < numberOfExamples; i = i + 1) {
        activationUnitsWithBias[i].unshift(1);
    }

    return activationUnitsWithBias;
};

var removeBias = function (deltaUnitsWithBias) {
    var deltaUnits = numeric.clone(deltaUnitsWithBias);
    for (i = 0; i < numberOfExamples; i = i + 1) {
        deltaUnits[i].shift();
    }
    return deltaUnits;
};

var computeActivationUnits = function (inputActivationUnits, ThetaTransposed){
    var outputActivationUnits = numeric.dot(inputActivationUnits, ThetaTransposed);
    outputActivationUnits = numeric.div(1, numeric.add(1, numeric.exp(numeric.mul(-1, outputActivationUnits))));

    return outputActivationUnits;
};

var computeActivationUnitsWithBias = function (activationUnitsWithBias, ThetaTransposed) {
    var activationUnits = computeActivationUnits(activationUnitsWithBias, ThetaTransposed);

    return addBias(activationUnits);
}

var computeDeltaUnitsWithBias = function (activationUnitsWithBias, DeltaUnits, Theta) {
    var sigmaGradient = numeric.mul(activationUnitsWithBias, numeric.sub(1, activationUnitsWithBias));
    var deltaUnitsWithBias = numeric.dot(DeltaUnits, Theta);

    deltaUnitsWithBias = numeric.mul(deltaUnitsWithBias, sigmaGradient);

    return deltaUnitsWithBias;
}

var computeDeltaUnits = function (activationUnitsWithBias, DeltaUnits, Theta) {
    var deltaUnitsWithBias = computeDeltaUnitsWithBias(activationUnitsWithBias, DeltaUnits, Theta);
    return removeBias(deltaUnitsWithBias);
}


var costFunction = function () {
    'use strict';

    var i;

    var Theta1;
    var Theta2;
    var Theta3;
    var Theta1Transposed;
    var Theta2Transposed;
    var Theta3Transposed;
    var theta1Vec;
    var theta2Vec;
    var theta3Vec;
    var A1withBias;
    var A2withBias;
    var A3withBias;
    var A4;
    var cost;
    var D2;
    var D3;
    var D4;
    var sigmaGradientLayer2;
    var sigmaGradientLayer3;
    var gtheta1Vec;
    var gtheta2Vec;
    var gtheta3Vec;
    var GradTheta1;
    var GradTheta2;
    var GradTheta3;
    var gradient;

    theta1Vec = ThetaVec.slice(0, (numberOfFeatures + 1) * numberOfActivationUnitsL1);
    theta2Vec = ThetaVec.slice((numberOfFeatures + 1) * numberOfActivationUnitsL1, (numberOfActivationUnitsL1 + 1) * (numberOfActivationUnitsL2) + (numberOfFeatures + 1) * numberOfActivationUnitsL1);
    theta3Vec = ThetaVec.slice((numberOfActivationUnitsL1 + 1) * (numberOfActivationUnitsL2) + (numberOfFeatures + 1) * numberOfActivationUnitsL1, ThetaVec.length);

    Theta1 = convertVectorToMatrix(theta1Vec, numberOfActivationUnitsL1, numberOfFeatures + 1);
    Theta2 = convertVectorToMatrix(theta2Vec, numberOfActivationUnitsL2, numberOfActivationUnitsL1 + 1);
    Theta3 = convertVectorToMatrix(theta3Vec, 1, numberOfActivationUnitsL2 + 1);

    GradTheta1 = numeric.rep(numeric.dim(Theta1), 0);
    GradTheta2 = numeric.rep(numeric.dim(Theta2), 0);
    GradTheta3 = numeric.rep(numeric.dim(Theta3), 0);

    Theta1Transposed = numeric.transpose(Theta1);
    Theta2Transposed = numeric.transpose(Theta2);
    Theta3Transposed = numeric.transpose(Theta3);

    A1withBias = addBias(numeric.clone(X));
    A2withBias = computeActivationUnitsWithBias(A1withBias, Theta1Transposed);
    A3withBias = computeActivationUnitsWithBias(A2withBias, Theta2Transposed);

    A4 = computeActivationUnits(A3withBias, Theta3Transposed);

    cost = 0;

    for (i = 0; i < numberOfExamples; i = i + 1) {
        cost += -1 * y[i][0] * Math.log(A4[i][0]) - ( 1 - y[i][0] ) * Math.log(1 - A4[i][0]);
    }

    cost = cost / numberOfExamples;

    D4 = numeric.sub(A4, y);
    D3 = computeDeltaUnits(A3withBias, D4, Theta3);
    D2 = computeDeltaUnits(A2withBias, D3, Theta2);

    for (i = 0; i < numberOfExamples; i = i + 1) {
        GradTheta1 = numeric.add(GradTheta1, numeric.tensor(D2[i], A1withBias[i]));
        GradTheta2 = numeric.add(GradTheta2, numeric.tensor(D3[i], A2withBias[i]));
        GradTheta3 = numeric.add(GradTheta3, numeric.tensor(D4[i], A3withBias[i]));
    }

    gtheta3Vec = numeric.mul(1 / numberOfExamples, convertMatrixToVector(GradTheta3));
    gtheta2Vec = numeric.mul(1 / numberOfExamples, convertMatrixToVector(GradTheta2));
    gtheta1Vec = numeric.mul(1 / numberOfExamples, convertMatrixToVector(GradTheta1));

    gradient = gtheta1Vec.concat(gtheta2Vec, gtheta3Vec);

    numberOfFeatures = null;
    numberOfActivationUnitsL1 = null;
    X = null;
    Theta1 = null;
    Theta2 = null;
    ThetaVec = null;
    theta1Vec = null;
    theta2Vec = null;
    A1withBias = null;
    A2withBias = null;
    A3withBias = null;
    D2 = null;
    D3 = null;
    sigmaGradientLayer2 = null;
    sigmaGradientLayer3 = null;
    gtheta1Vec = null;
    gtheta2Vec = null;
    gtheta3Vec = null;

    return [cost, gradient, A4, y];
};

process.on('message', function (setup) {
    'use strict';

    numberOfFeatures = setup.numberOfFeatures;
    numberOfActivationUnitsL1 = setup.numberOfActivationUnitsL1;
    numberOfActivationUnitsL2 = setup.numberOfActivationUnitsL2;

    X = setup.X;
    y = setup.Y;

    numberOfExamples = X.length;

    ThetaVec = setup.ThetaVec;

    process.send(costFunction());
});
