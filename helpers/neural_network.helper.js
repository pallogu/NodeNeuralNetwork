/*global process:true*/

(function () {
    var la = require('./linear_algebra.helper.js');
    var Theta1;
    var Theta2;
    var Theta3;
    var trainingSetInput;
    var trainingSetOutput;
    var lambda;
    var a1WithBias;
    var z2;
    var a2;
    var a2WithBias;
    var z3;
    var a3;
    var a3WithBias;
    var z4;
    var a4;
    var Theta1WithZeros;
    var Theta2WithZeros;
    var Theta3WithZeros;
    var cost;
    var d3;
    var d2;
    var d1;
    var D1;
    var D2;
    var D3;
    var exampleIndex;
    var l;
    var trainingResult;

    var computeGradientsWithCost = function (options) {

        Theta1 = options.Theta1;
        Theta2 = options.Theta2;
        Theta3 = options.Theta3;
        trainingSetInput = options.trainingSetInput;
        trainingSetOutput = options.trainingSetOutput;
        lambda = options.lambda;

        a1WithBias = la.addBias(trainingSetInput);
        z2 = la.computeZ(Theta1, a1WithBias);
        a2 = la.computeA(z2);
        a2WithBias = la.addBias(a2);
        z3 = la.computeZ(Theta2, a2WithBias);
        a3 = la.computeA(z3);
        a3WithBias = la.addBias(a3);
        z4 = la.computeZ(Theta3,a3WithBias);
        a4 = la.computeA(z4);

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

        return {
            cost: cost,
            prediction: a4,
            D1: D1,
            D2: D2,
            D3: D3
        };
    };


    process.on('message', function (setup) {
        'use strict';

        trainingResult = computeGradientsWithCost({
            Theta1: setup.Theta1,
            Theta2: setup.Theta2,
            Theta3: setup.Theta3,
            trainingSetInput: setup.X,
            trainingSetOutput: setup.Y,
            lambda: setup.lambda
        });

        process.send(trainingResult);
    });

})();