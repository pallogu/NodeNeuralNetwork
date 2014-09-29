var assert = require('assert');
var _ = require('lodash');
var M = require('eigenjs').Matrix;
var fixture = require('./fixtures/nn_helper.fix.js');
var NN_helper = require('../helpers/nn_helper.js');
var expect = require('chai').expect;

var convertMatrixToArray = function (matrix) {
//    var cleanfmt = {
//        coeffSeparator: ','
//        , matPrefix: '['
//        , matSuffix: ']'
//        , rowPrefix: '['
//        , rowSuffix: '],'
//    };

    var cleanfmt = {
        precision: 16,
        matPrefix: '[',
        matSuffix: ']',
        dontAlignCols:true,
        coeffSeparator: ',',
        rowPrefix: '[',
        rowSuffix: '],'
    };

    var s = matrix.toString(cleanfmt).slice(0, -2) + ']';
    s = JSON.parse(s);
    return s;
};


describe('NN Helper', function () {

    var trainingSetInputMat = new M(4,2);
    trainingSetInputMat.set(_.flatten(fixture.trainingSetInput));

    var trainingSetOutputMat = new M(4,1);
    trainingSetOutputMat.set(_.flatten(fixture.trainingSetOutput));


    var Theta1Mat = new M(2, 3);
    var Theta2Mat = new M(2, 3);
    var Theta3Mat = new M(1, 3);

    Theta1Mat.set(_.flatten(fixture.Theta1));
    Theta2Mat.set(_.flatten(fixture.Theta2));
    Theta3Mat.set(_.flatten(fixture.Theta3));

    var nn_helper;

    beforeEach(function () {
        nn_helper = new NN_helper({
            Theta1: Theta1Mat,
            Theta2: Theta2Mat,
            Theta3: Theta3Mat,
            trainingSetInput: trainingSetInputMat,
            trainingSetOutput: trainingSetOutputMat,
            lamda: fixture.lamda
        });
    });

    describe('addBias method', function () {
        it('should add 1  to a column', function () {
            var a0WithBiasMat  = nn_helper.addBias(trainingSetInputMat);
            expect(convertMatrixToArray(a0WithBiasMat)).to.deep.equal(fixture.a0WithBias);
        });
    });

    describe('computeZ method', function () {
        it('should compute next layer activation units before logistic reg is applied', function () {
            var a0WithBiasMat = nn_helper.addBias(trainingSetInputMat);
            var z1Mat = nn_helper.computeZ(a0WithBiasMat, Theta1Mat);

            expect(convertMatrixToArray(z1Mat)).to.deep.equal(fixture.z1);
        });
    });

    describe('computeA method', function () {
        it('should compute logistic regression on activation units', function () {
            var a0WithBiasMat = nn_helper.addBias(trainingSetInputMat);
            var z1Mat = nn_helper.computeZ(a0WithBiasMat, Theta1Mat);
            var a1Mat = nn_helper.computeA(z1Mat);

            expect(convertMatrixToArray(a1Mat)).to.deep.equal(fixture.a1);
        });
    });

    describe('cost', function () {
        it.only('should return cost for all training set inputs', function () {
            var cost = nn_helper.cost;
            expect(cost).to.equal(fixture.cost);
        });
    });
});