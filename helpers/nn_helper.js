'use strict';
var _ = require('underscore');
var M = require('eigenjs').Matrix;

var NN_helper = function (options) {

    var Theta1 = options.Theta1;
    var Theta2 = options.Theta2;
    var Theta3 = options.Theta3;
    var trainingSetInput = options.trainingSetInput;
    var trainingSetOutput = options.trainingSetOutput;
    var lambda = options.lamda;

    var addBias = function (mat) {
        var matWithBias = new M.Ones(mat.rows(), mat.cols() + 1);
        var block = matWithBias.block(0,1,mat.rows(), mat.cols());
        block.assign(mat);
        return matWithBias;
    };

    var computeZ = function (aWithBias, Theta) {
        var z = aWithBias.mul(Theta.transpose());
        return z;
    };

    var computeA = function (zMat) {
        var a = new M(zMat.rows(), zMat.cols());
        zMat.visit(function (z, r, c) {
            a.set(r, c, 1 / ( 1  + Math.exp(-1 * z)));
        });

        return a;
    };

    var sumThetaSquared = function (mat) {
        var sumSquared = 0;

        mat.visit(function (val) {
            sumSquared += val*val;
        });

        return sumSquared;
    };
    
    var a1WithBias = addBias(trainingSetInput);
    var z2 = computeZ(a1WithBias, Theta1);
    var a2 = computeA(z2);
    var a2WithBias = addBias(a2);
    var z3 = computeZ(a2WithBias, Theta2);
    var a3 = computeA(z3);
    var a3WithBias = addBias(a3);
    var z4 = computeZ(a3WithBias, Theta3);
    var a4 = computeA(z4);

    var computeCost = function() {
        var sum = 0;
        
        a4.visit(function (value, r, c) {
            sum += trainingSetOutput.get(r, c) * Math.log(a4.get(r, c));
            sum += (1-trainingSetOutput.get(r, c)) * Math.log(1 - a4.get(r, c));
        });

        sum = -1 * sum;
        sum += sumThetaSquared(Theta1)*lambda/2;
        sum += sumThetaSquared(Theta2)*lambda/2;
        sum += sumThetaSquared(Theta3)*lambda/2;

        return sum;
    };
    
    var cost = computeCost();

    var d3 = a4.sub(trainingSetOutput);

    var computeD = function (Theta, d, a) {
        var scalar;
        var tmpMat = d.mul(Theta);
        tmpMat = tmpMat.rightCols(a.cols());

        tmpMat.visit(function (value, row, col) {
            scalar = value * a.get(row, col) * (1 - a.get(row, col));
            tmpMat.set(row, col, scalar);
            scalar = undefined;
        });

        return tmpMat;
    };

    var d2 = computeD(Theta3, d3, a3);
    var d1 = computeD(Theta2, d2, a2);

    var D1 = M.Zero(Theta1.rows(), Theta1.cols());
    var D2 = M.Zero(Theta2.rows(), Theta2.cols());
    var D3 = M.Zero(Theta3.rows(), Theta3.cols());

    for (var exampleIndex = 0, l = trainingSetInput.rows(); exampleIndex < l; exampleIndex++) {
        D1 = D1.add(d1.row(exampleIndex).transpose().mul(a1WithBias.row(exampleIndex)));
        D2 = D2.add(d2.row(exampleIndex).transpose().mul(a2WithBias.row(exampleIndex)));
        D3 = D3.add(d3.row(exampleIndex).transpose().mul(a3WithBias.row(exampleIndex)));
    }

    var setFirstColumnToZeros = function (mat) {
        var tmpMat = M.Zero(mat.rows(), mat.cols());
        tmpMat.rightCols(mat.cols()-1).assign(mat.rightCols(mat.cols()-1));

        return tmpMat;
    };

    var Theta1WithZeros = setFirstColumnToZeros(Theta1);
    var Theta2WithZeros = setFirstColumnToZeros(Theta2);
    var Theta3WithZeros = setFirstColumnToZeros(Theta3);

    D1 = D1.add(Theta1WithZeros.mul(lambda));
    D2 = D2.add(Theta2WithZeros.mul(lambda));
    D3 = D3.add(Theta3WithZeros.mul(lambda));

    var computeCost = function () {
        return {
            cost: cost,
            prediction: a4,
            D1: D1,
            D2: D2,
            D3: D3
        }
    };

    return {
        addBias : addBias,
        computeA: computeA,
        computeZ :  computeZ,
        computeCost: computeCost()
    };
};

module.exports = NN_helper;