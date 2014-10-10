var _ = require('lodash');

//nr - number of rows;
//nc - number of columns;

var LinearAlgebraHelper = function LinearAlgebraHelper () {};

_.extend(LinearAlgebraHelper.prototype, {
    clone2dMatrix: function (mat) {
        var nr = mat.length;
        var tmpMat = new Array(nr);
        var r = 0;

        for(r = 0; r < nr; r++) {
            tmpMat[r] = mat[r].slice(0);
        }

        nr = null;
        r = null;

        return tmpMat;
    },
    random2DMatrix: function (nr, nc, epsilon) {
        var e =  epsilon || 1;
        var tmpMat = new Array(nr);
        var r = 0;
        var c = 0;

        for(r = 0; r < nr; r++) {
            tmpMat[r] = new Array(nc);
            for(c = 0; c < nc; c++) {
                tmpMat[r][c] = Math.random()*2*e - e;
            }
        }

        e = null;
        r = null;
        c = null;

        return tmpMat;
    },
    add2DMatrices :  function (mat1, mat2) {
        var nr = mat1.length;
        var nc = mat1[0].length;
        var r = 0;
        var c = 0;
        var tmpMat = new Array(nr);

        for(r = 0; r < nr; r++) {
            tmpMat[r] = new Array(nc);
            for(c = 0; c < nc; c++) {
                tmpMat[r][c] = mat1[r][c] + mat2[r][c];
            }
        }

        nr = null;
        nc = null;
        r = null;
        c = null;

        return tmpMat;
    },
    sub2DMatrices: function (mat1, mat2) {
        var nr = mat1.length;
        var nc = mat1[0].length;
        var r = 0;
        var c = 0;
        var tmpMat = new Array(nr);

        for(r = 0; r < nr; r++) {
            tmpMat[r] = new Array(nc);
            for(c = 0; c < nc; c++) {
                tmpMat[r][c] = mat1[r][c] - mat2[r][c];
            }
        }

        nr = null;
        nc = null;
        r = null;
        c = null;

        return tmpMat;
    },
    mul2DMatrixByScalar: function (mat, s) {
        var nr = mat.length;
        var nc = mat[0].length;
        var r = 0;
        var c = 0;
        var tmpMat = new Array(nr);

        for(r = 0; r < nr; r++) {
            tmpMat[r] = new Array(nc);
            for(c = 0; c < nc; c++) {
                tmpMat[r][c] = mat[r][c] * s;
            }
        }

        nr = null;
        nc = null;
        r = null;
        c = null;

        return tmpMat;
    },
    addBias: function (mat) {
        var nr = mat.length;
        var r = 0;
        var tmpMat = new Array(nr);

        for(r = 0; r < nr; r++) {
            tmpMat[r] = mat[r].slice(0);
            tmpMat[r].unshift(1);
        }

        nr = null;
        r = null;

        return tmpMat;
    },
    computeZ: function (Theta, AWB) {
        var nr = AWB.length;
        var r = 0;
        var nc = Theta.length;
        var c = 0;
        var counterLength = AWB[0].length;
        var counter = 0;
        var sum = 0;

        var tmpMat = new Array(nr);

        for(r = 0; r < nr; r++) {
            tmpMat[r] = new Array(nc);
            for (c = 0; c < nc; c++) {
                sum = 0;
                for(counter = 0; counter < counterLength; counter++) {
                    sum += AWB[r][counter] * Theta[c][counter];
                }
                tmpMat[r][c] = sum;
            }
        }

        nr = null;
        r = null;
        nc = null;
        c = null;
        counterLength = null;
        counter = null;
        sum = null;

        return tmpMat;
    },
    computeA: function (Z) {
        var nr = Z.length;
        var r = 0;
        var nc = Z[0].length;
        var c = 0;

        var tmpMat = new Array(nr);

        for(r = 0; r < nr; r++) {
            tmpMat[r] = new Array(nc);
            for (c = 0; c < nc; c++) {
                tmpMat[r][c] = 1 / (1 + Math.exp(-1 * Z[r][c]));
            }
        }

        nr = null;
        r = null;
        nc = null;
        c = null;

        return tmpMat;
    },
    sumOfSquares: function (mat) {
        var nr = mat.length;
        var r = 0;
        var nc = mat[0].length;
        var c = 0;

        var sum = 0;

        for(r = 0; r < nr; r++) {
            for (c = 0; c < nc; c++) {
                sum += mat[r][c] * mat[r][c];
            }
        }

        nr = null;
        r = null;
        nc = null;
        c = null;

        return sum;
    },
    setFirstColumnToZeros: function (mat) {
        var nr = mat.length;
        var tmpMat = new Array(nr);
        var r = 0;

        for(r = 0; r < nr; r++) {
            tmpMat[r] = mat[r].slice(0);
            tmpMat[r][0] = 0;
        }

        nr = null;
        r = null;

        return tmpMat;
    }
});

module.exports = new LinearAlgebraHelper();