/*global process:true*/
var _ = require('lodash');
var M = require('eigenjs').Matrix;
var NN_helper = require('./nn_helper.js');

var convertToArray = function (mat) {
    var cleanfmt = {
        precision: 16,
        matPrefix: '[',
        matSuffix: ']',
        dontAlignCols:true,
        coeffSeparator: ',',
        rowPrefix: '',
        rowSuffix: ','
    };

    var s = mat.toString(cleanfmt).slice(0, -2) + ']';
    s = JSON.parse(s);
    return s;
};

process.on('message', function (setup) {
    'use strict';

    var Theta1 = new M(setup.Theta1.rows, setup.Theta1.cols);
    var Theta2 = new M(setup.Theta2.rows, setup.Theta2.cols);
    var Theta3 = new M(setup.Theta3.rows, setup.Theta3.cols);

    Theta1.set(setup.Theta1.vector);
    Theta2.set(setup.Theta2.vector);
    Theta3.set(setup.Theta3.vector);

    var XMat = new M(setup.X.length, setup.X[0].length);
    var YMat = new M(setup.Y.length, setup.Y[0].length);

    XMat.set(_.flatten(setup.X));
    YMat.set(_.flatten(setup.Y));

    var nn_helper = new NN_helper({
        Theta1: Theta1,
        Theta2: Theta2,
        Theta3: Theta3,
        trainingSetInput: XMat,
        trainingSetOutput: YMat,
        lambda: setup.lambda
    });

    var trainingResult = nn_helper.result;

    trainingResult.prediction = convertToArray(trainingResult.prediction)[0];
    trainingResult.D1 = convertToArray(trainingResult.D1);
    trainingResult.D2 = convertToArray(trainingResult.D2);
    trainingResult.D3 = convertToArray(trainingResult.D3);

    process.send(trainingResult);
});
