/*global process:true*/
(function () {


    var _ = require('lodash');
    var M = require('eigenjs').Matrix;

    var tmpMat;
    var sum = 0;

    var convertToArray = function (mat) {
        return JSON.parse(mat.toString({
            precision: 16,
            matPrefix: '[',
            matSuffix: ']',
            dontAlignCols:true,
            coeffSeparator: ',',
            rowPrefix: '',
            rowSuffix: ','
        }).slice(0, -2) + ']');
    };

    var addBias = function (mat) {
        tmpMat = new M.Ones(mat.rows(), mat.cols() + 1);
        var block = tmpMat.block(0,1,mat.rows(), mat.cols());
        block.assign(mat);
        return tmpMat;
    };

    var computeZ = function (aWithBias, Theta) {
        return aWithBias.mul(Theta.transpose());
    };

    var computeA = function (zMat) {
        tmpMat = new M(zMat.rows(), zMat.cols());
        zMat.visit(function (z, r, c) {
            tmpMat.set(r, c, 1 / ( 1  + Math.exp(-1 * z)));
        });

        return tmpMat;
    };

    var sumThetaSquared = function (mat) {
        sum = 0;

        mat.visit(function (val) {
            sum += val*val;
        });

        return sum;
    };

    var setFirstColumnToZeros = function (mat) {
        tmpMat = M.Zero(mat.rows(), mat.cols());
        tmpMat.rightCols(mat.cols()-1).assign(mat.rightCols(mat.cols()-1));

        return tmpMat;
    };

    var computeD = function (Theta, d, a) {
        tmpMat = d.mul(Theta);
        tmpMat = tmpMat.rightCols(a.cols());

        tmpMat.visit(function (value, row, col) {
            sum = value * a.get(row, col) * (1 - a.get(row, col));
            tmpMat.set(row, col, sum);
        });

        return tmpMat;
    };

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
    var predictionArray;
    var D1Array;
    var D2Array;
    var D3Array;
    var Theta1FromMaster;
    var Theta2FromMaster;
    var Theta3FromMaster;
    var XMat;
    var YMat;
    var trainingResult;

    var computeGradientsWithCost = function (options) {

        Theta1 = options.Theta1;
        Theta2 = options.Theta2;
        Theta3 = options.Theta3;
        trainingSetInput = options.trainingSetInput;
        trainingSetOutput = options.trainingSetOutput;
        lambda = options.lambda;

        a1WithBias = addBias(trainingSetInput);
        z2 = computeZ(a1WithBias, Theta1);
        a2 = computeA(z2);
        a2WithBias = addBias(a2);
        z3 = computeZ(a2WithBias, Theta2);
        a3 = computeA(z3);
        a3WithBias = addBias(a3);
        z4 = computeZ(a3WithBias, Theta3);
        a4 = computeA(z4);

        Theta1WithZeros = setFirstColumnToZeros(Theta1);
        Theta2WithZeros = setFirstColumnToZeros(Theta2);
        Theta3WithZeros = setFirstColumnToZeros(Theta3);


        cost = (function() {
            sum = 0;

            a4.visit(function (value, r, c) {
                sum += trainingSetOutput.get(r, c) * Math.log(a4.get(r, c));
                sum += (1-trainingSetOutput.get(r, c)) * Math.log(1 - a4.get(r, c));
            });

            sum = -1 * sum;
            if (lambda) {
                sum += sumThetaSquared(Theta1WithZeros)*lambda/2;
                sum += sumThetaSquared(Theta2WithZeros)*lambda/2;
                sum += sumThetaSquared(Theta3WithZeros)*lambda/2;
            }

            return sum;
        }());

        d3 = a4.sub(trainingSetOutput);
        d2 = computeD(Theta3, d3, a3);
        d1 = computeD(Theta2, d2, a2);

        D1 = M.Zero(Theta1.rows(), Theta1.cols());
        D2 = M.Zero(Theta2.rows(), Theta2.cols());
        D3 = M.Zero(Theta3.rows(), Theta3.cols());
        exampleIndex = 0;
        l = trainingSetInput.rows();
        for (exampleIndex = 0 ; exampleIndex < l; exampleIndex++) {
            D1 = D1.add(d1.row(exampleIndex).transpose().mul(a1WithBias.row(exampleIndex)));
            D2 = D2.add(d2.row(exampleIndex).transpose().mul(a2WithBias.row(exampleIndex)));
            D3 = D3.add(d3.row(exampleIndex).transpose().mul(a3WithBias.row(exampleIndex)));
        }

        if(lambda) {
            D1 = D1.add(Theta1WithZeros.mul(lambda));
            D2 = D2.add(Theta2WithZeros.mul(lambda));
            D3 = D3.add(Theta3WithZeros.mul(lambda));
        }

        predictionArray = convertToArray(a4);
        D1Array = convertToArray(D1);
        D2Array = convertToArray(D2);
        D3Array = convertToArray(D3);

        return {
            cost: cost,
            prediction: predictionArray,
            D1: D1Array,
            D2: D2Array,
            D3: D3Array
        };
    };


    process.on('message', function (setup) {
        'use strict';
        Theta1FromMaster = new M(setup.Theta1.rows, setup.Theta1.cols);
        Theta2FromMaster = new M(setup.Theta2.rows, setup.Theta2.cols);
        Theta3FromMaster = new M(setup.Theta3.rows, setup.Theta3.cols);

        Theta1FromMaster.set(setup.Theta1.vector);
        Theta2FromMaster.set(setup.Theta2.vector);
        Theta3FromMaster.set(setup.Theta3.vector);

        XMat = new M(setup.X.length, setup.X[0].length);
        YMat = new M(setup.Y.length, setup.Y[0].length);

        XMat.set(_.flatten(setup.X));
        YMat.set(_.flatten(setup.Y));

        trainingResult = computeGradientsWithCost({
            Theta1: Theta1FromMaster,
            Theta2: Theta2FromMaster,
            Theta3: Theta3FromMaster,
            trainingSetInput: XMat,
            trainingSetOutput: YMat,
            lambda: setup.lambda
        });

//        console.log(trainingResult);
        process.send(trainingResult);
    });

})();