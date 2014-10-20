/*global process:true*/
'use strict';
(function () {

//    var La = require('./linear_algebra.helper.js');
    var la = require('./linear_algebra.helper.js');

//    var memwatch = require('memwatch');

//    memwatch.on('leak', function(info) {
//        console.log(info);
//    });



    var computeGradientsWithCost = function (setup) {

        var parsedSetup = JSON.parse(setup);

        var Theta1 = parsedSetup.Theta1;
        var Theta2 = parsedSetup.Theta2;
        var Theta3 = parsedSetup.Theta3;
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
        var result = JSON.stringify({
            cost: cost,
            prediction: a4,
            D1: D1,
            D2: D2,
            D3: D3
        });

        return result
    };


    process.on('message', function (setup) {
        'use strict';
        process.send(computeGradientsWithCost(setup));
    });

})();