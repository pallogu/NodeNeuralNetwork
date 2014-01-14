/*global process:true*/
var numeric = require('numeric');
var _ = require('underscore');

var numberOfFeatures;
var numberOfActivationUnitsL1;
var numberOfActivationUnitsL2;
var X;
var y;
var ThetaVec;

var convertMatrictToVector = function (matrix) {
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
    var A1;
    var A1withBias;
    var A2;
    var A2withBias;
    var A3;
    var A3withBias;
    var A4;
    var cost;
    var D2;
    var D2withBias;
    var D3;
    var D3withBias;
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
    var numberOfExamples = X.length;

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


    A1 = numeric.clone(X);
    A1withBias = numeric.clone(A1);

    for (i = 0; i < numberOfExamples; i = i + 1) {
        A1withBias[i].unshift(1);
    }

    A2 = numeric.dot(A1withBias, Theta1Transposed);
    A2 = numeric.div(1, numeric.add(1, numeric.exp(numeric.mul(-1, A2))));

    A2withBias = numeric.clone(A2);
    for (i = 0; i < numberOfExamples; i = i + 1) {
        A2withBias[i].unshift(1);
    }

    A3 = numeric.dot(A2withBias, Theta2Transposed);
    A3 = numeric.div(1, numeric.add(1, numeric.exp(numeric.mul(-1, A3))));

    A3withBias = numeric.clone(A3);
    for (i = 0; i < numberOfExamples; i = i + 1) {
        A3withBias[i].unshift(1);
    }

    A4 = numeric.dot(A3withBias, Theta3Transposed);
    A4 = numeric.div(1, numeric.add(1, numeric.exp(numeric.mul(-1, A4))));


    cost = 0;

    for (i = 0; i < numberOfExamples; i = i + 1) {
        cost += -1 * y[i][0] * Math.log(A4[i][0]) - ( 1 - y[i][0] ) * Math.log(1 - A4[i][0]);
    }

    cost = cost / numberOfExamples;

    D4 = numeric.sub(A4, y);

    sigmaGradientLayer3 = numeric.mul(A3withBias, numeric.sub(1, A3withBias));
    D3withBias = numeric.dot(D4, Theta3);

    D3withBias = numeric.mul(D3withBias, sigmaGradientLayer3);

    D3 = numeric.clone(D3withBias);

    for (i = 0; i < numberOfExamples; i = i + 1) {
        D3[i].shift();
    }

    sigmaGradientLayer2 = numeric.mul(A2withBias, numeric.sub(1, A2withBias));
    D2withBias = numeric.dot(D3, Theta2);


    D2withBias = numeric.mul(D2withBias, sigmaGradientLayer2);
    D2 = numeric.clone(D2withBias);

    for (i = 0; i < numberOfExamples; i = i + 1) {
        D2[i].shift();
    }

    for (i = 0; i < numberOfExamples; i = i + 1) {
        GradTheta1 = numeric.add(GradTheta1, numeric.tensor(D2[i], A1withBias[i]));
        GradTheta2 = numeric.add(GradTheta2, numeric.tensor(D3[i], A2withBias[i]));
        GradTheta3 = numeric.add(GradTheta3, numeric.tensor(D4[i], A3withBias[i]));
    }

    gtheta3Vec = numeric.mul(1 / numberOfExamples, convertMatrictToVector(GradTheta3));
    gtheta2Vec = numeric.mul(1 / numberOfExamples, convertMatrictToVector(GradTheta2));
    gtheta1Vec = numeric.mul(1 / numberOfExamples, convertMatrictToVector(GradTheta1));

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
    A2 = null;
    A2withBias = null;
    A3 = null;
    A3withBias = null;
    D2 = null;
    D2withBias = null;
    D3 = null;
    D3withBias = null;
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

    ThetaVec = setup.ThetaVec;

    process.send(costFunction());
});
