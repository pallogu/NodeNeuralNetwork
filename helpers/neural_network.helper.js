/*global process:true*/
var _ = require('underscore');
var NN_helper = require('./nn_helper.js');



process.on('message', function (setup) {
    'use strict';

    var nn_helper = new NN_helper({
        Theta1: setup.Theta1,
        Theta2: setup.Theta2,
        Theta3: setup.Theta3,
        trainingSetInput: setup.X,
        trainingSetOutput: setup.Y,
        lambda: setup.lambda
    });

    process.send(nn_helper.computeCost());
});
