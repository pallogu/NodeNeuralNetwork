/*global process:true*/
var La = require('./linear_algebra.helper.js');
var _ = require('lodash');


var computeGradientsWithCost = function (setup) {

    var parsedSetup = JSON.parse(setup);

    var la = new La();

    var Theta1 = parsedSetup.Theta1;
    var Theta2 = parsedSetup.Theta2;
    var Theta3 = parsedSetup.Theta3;

    var batchSize = parsedSetup.batchSize;
    var trainingSetInput = parsedSetup.X;
    var trainingSetOutput = parsedSetup.Y;
    var lambda = parsedSetup.lambda;
    var a1WithBias;
    var a2WithBias;
    var a3WithBias;
    var a4;
    var Theta1WithZeros;
    var Theta2WithZeros;
    var Theta3WithZeros;
    var cost;
    var d3;
    var d2;
    var d1;
    var d;
    var D1;
    var D2;
    var D3;
    var exampleIndex;
    var l;

    a1WithBias = la.addBias(trainingSetInput);
    a2WithBias = la.addBias(la.computeA(la.computeZ(Theta1, a1WithBias)));
    a3WithBias = la.addBias(la.computeA(la.computeZ(Theta2, a2WithBias)));
    a4 = la.computeA(la.computeZ(Theta3,a3WithBias));

    Theta1WithZeros = la.setFirstColumnToZeros(Theta1);
    Theta2WithZeros = la.setFirstColumnToZeros(Theta2);
    Theta3WithZeros = la.setFirstColumnToZeros(Theta3);
    cost = la.computeCost(a4, trainingSetOutput);

    if (lambda) {
        cost += la.sumOfSquares(Theta1WithZeros)*lambda/2;
        cost += la.sumOfSquares(Theta2WithZeros)*lambda/2;
        cost += la.sumOfSquares(Theta3WithZeros)*lambda/2;
    }

    d3 = la.sub2DMatrices(a4,trainingSetOutput);
    d2 = la.computeD(Theta3, d3, a3WithBias);
    d1 = la.computeD(Theta2, d2, a2WithBias);

    D1 = la.create2DMatrix(Theta1.length, Theta1[0].length);
    D2 = la.create2DMatrix(Theta2.length, Theta2[0].length);
    D3 = la.create2DMatrix(Theta3.length, Theta3[0].length);
    l = trainingSetInput.length;

    for (exampleIndex = 0 ; exampleIndex < l; exampleIndex++) {
        D1 = la.add2DMatrices(D1, la.computeDTensorSlice(a1WithBias[exampleIndex], d1[exampleIndex]));
        D2 = la.add2DMatrices(D2, la.computeDTensorSlice(a2WithBias[exampleIndex], d2[exampleIndex]));
        D3 = la.add2DMatrices(D3, la.computeDTensorSlice(a3WithBias[exampleIndex], d3[exampleIndex]));
    }

    if(lambda) {
        D1 = la.add2DMatrices(D1, la.mul2DMatrixByScalar(Theta1WithZeros,lambda));
        D2 = la.add2DMatrices(D2, la.mul2DMatrixByScalar(Theta2WithZeros,lambda));
        D3 = la.add2DMatrices(D3, la.mul2DMatrixByScalar(Theta3WithZeros,lambda));
    }

    D1 = la.mul2DMatrixByScalar(D1, 1/batchSize);
    D2 = la.mul2DMatrixByScalar(D2, 1/batchSize);
    D3 = la.mul2DMatrixByScalar(D3, 1/batchSize);

    cost = cost / batchSize;

    var result = JSON.stringify({
        cost: cost,
        prediction: a4,
        D1 : D1,
        D2 : D2,
        D3 : D3
    });

    return result
};


process.on('message', function (setup) {
    'use strict';
//        var foo = '{"cost":1.3943109806507783,"prediction":[[0.5455325878886108],[0.5453913066364938]],"D1":[[-0.0002444293228990623,-0.0002444293228990623,-0.0018191626804777736],[0.00035026699471869555,0.00035026699471869555,0.002733335528312702],[0.00142104480117512,0.00142104480117512,0.008642844868422253],[0.0001356228119276518,0.0001356228119276518,0.0006879020472877486]],"D2":[[0.010376532929863978,0.009917341162240797,0.009940413909217356,0.004076297944780798,0.0049255240968755325],[-0.0033451683608757728,-0.0032424046135614854,-0.0032497615352375716,-0.0013043662115295087,-0.0015741266048676684],[-0.009245672781343355,-0.0088888172628428,-0.008909281653985045,-0.0036207979025071196,-0.004372839333176563],[0.009941975148329604,0.009509235517237119,0.009531329044702955,0.003904032727376359,0.0047170542628379546]],"D3":[[0.09092389452510463,0.054227221090903666,0.032857598364170876,0.03747631959166148,0.04593299757399083]]}';
    var result = computeGradientsWithCost(setup);
//        console.log();
    process.send(result);
    result = null;
    setup = null;
    computeGradientsWithCost = null;
    La = null;
    _ = null;
});