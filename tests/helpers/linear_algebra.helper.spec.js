var expect = require('chai').expect;
var la = require('../../helpers/linear_algebra.helper.js');

describe('Linear Algebra Helper', function () {
    it('instantiation', function () {
        expect(la.constructor.name).to.equal('LinearAlgebraHelper');
    });

    describe('Clone 2d matrix method', function () {
        it('should make a copy of matrix', function () {
            var M1 = [[11,12],[21,22]];

            var M2 = la.clone2dMatrix(M1);

            expect(M1).not.to.equal(M2);
            expect(M1).to.deep.equal(M2);
            expect(M1[0]).not.to.equal(M2[0]);
            expect(M1[0]).to.deep.equal(M2[0]);
        });

    });

    describe('GenerateRandom2DMatrix', function () {
        it('should create a matrix with random values within -1 and 1 range if no epsilon is present', function () {
            var M1 = la.random2DMatrix(2, 2);
            var values = [];

            expect(M1.length).to.equal(2);
            expect(M1[0].length).to.equal(2);

            for (var r = 0; r < M1.length; r++) {
                for (var c = 0; c < M1[r].length; c++) {
                    expect(M1[r][c] < 1 && M1[r][c] > -1).to.equal(true);
                }
            }

        });

        it('should create a matrix with random values within -epsilon and epsilon range if epsilon is present', function () {
            var epsilon = 0.1;
            var M1 = la.random2DMatrix(2, 2, epsilon);
            var values = [];

            expect(M1.length).to.equal(2);
            expect(M1[0].length).to.equal(2);

            for (var r = 0; r < M1.length; r++) {
                for (var c = 0; c < M1[r].length; c++) {
                    expect(M1[r][c] < epsilon && M1[r][c] > -1*epsilon).to.equal(true);
                }
            }

        });
    });

    describe('add2DMatrices', function () {
        it('should add two matrices', function () {
            var M1 = [[11,12],[21,22]];
            var M2 = [[31,32],[41,42]];
            var expectedMatrix = [[42,44],[62,64]];

            var M3 = la.add2DMatrices(M1, M2);
            expect(M3).to.deep.equal(expectedMatrix);
        });
    });

    describe('sub2DMatrices', function () {
        it('should substract two matrices', function () {
            var M1 = [[11,12],[21,22]];
            var M2 = [[33,35],[45,47]];
            var expectedMatrix = [[22,23],[24,25]];

            var M3 = la.sub2DMatrices(M2, M1);
            expect(M3).to.deep.equal(expectedMatrix);
        });
    });

    describe('mul2DMatrixByScalar', function () {
        it('should multiply every item in matrix by scalar', function () {
            var M1 = [[11, 12],[21, 22]];
            var scalar = 2;
            var expectedMatrix = [[22, 24],[42, 44]];

            var M2 = la.mul2DMatrixByScalar(M1, scalar);

            expect(M2).to.deep.equal(expectedMatrix);
        });
    });

    describe('addBias', function () {
        it('should add 1 in the beginning of each row', function () {
            var M1 = [[11, 12],[21, 22]];
            var expectedMatrix = [[1, 11, 12],[1, 21, 22]];

            var M2 = la.addBias(M1);

            expect(M2).to.deep.equal(expectedMatrix);
        });
    });

    describe('computeZ', function () {
        it('should multiply Theta and aWithBias', function () {
            var awb = [[1, 11, 12],[1, 21, 22],[1, 31, 32]];
            var Theta = [[1, 1.1, 1.2],[2, 2.1, 2.2]];
            var expectedMatrix = [
                [1*1 + 1.1*11 + 1.2 * 12, 2*1 + 2.1*11 + 2.2 * 12],
                [1*1 + 1.1*21 + 1.2 * 22, 2*1 + 2.1*21 + 2.2 * 22],
                [1*1 + 1.1*31 + 1.2 * 32, 2*1 + 2.1*31 + 2.2 * 32]
            ];

            var Z = la.computeZ(Theta,awb);

            expect(Z).to.deep.equal(expectedMatrix);
        });
    });

    describe('computeA', function () {
        it('should compute logistic function on each element of matrix', function () {
            var Z = [[11, 12],[21, 22],[31, 32]];
            var expectedMatrix = [
                [1 / ( 1  + Math.exp(-1 * 11)), 1 / ( 1  + Math.exp(-1 * 12))],
                [1 / ( 1  + Math.exp(-1 * 21)), 1 / ( 1  + Math.exp(-1 * 22))],
                [1 / ( 1  + Math.exp(-1 * 31)), 1 / ( 1  + Math.exp(-1 * 32))]
            ];

            var A = la.computeA(Z);

            expect(A).to.deep.equal(expectedMatrix);
        });
    });

    describe('sumOfSquares', function () {
        it('should return the sum of every element squared', function () {
            var Z = [[11, 12],[21, 22],[31, 32]];
            var expectedSum = 11*11 + 12*12 + 21*21 + 22*22 + 31*31 + 32*32;

            var sum = la.sumOfSquares(Z);

            expect(sum).to.equal(expectedSum);
        });
    });

    describe('setFirstColumnToZeros', function () {
        it('should return a matrix with first column set to zeros', function () {
            var mat = [[11, 12, 13],[21, 22, 23],[31, 32, 33]];
            var exptectedMatrix = [[0, 12, 13],[0, 22, 23],[0, 32, 33]];

            var matWithZeros = la.setFirstColumnToZeros(mat);

            expect(matWithZeros).to.deep.equal(exptectedMatrix);
        });
    });
});