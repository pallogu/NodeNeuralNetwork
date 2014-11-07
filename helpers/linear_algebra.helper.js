'use strict';

var LinearAlgebraHelper = function LinearAlgebraHelper() {};



LinearAlgebraHelper.prototype.create2DMatrix =  function (nr, nc) {
    var tmpMat = new Array(nr);
    var r = 0;
    var c = 0;
    for (r = 0; r < nr; r++) {
        tmpMat[r] = new Array(nc);
        for (c = 0; c < nc; c++) {
            tmpMat[r][c] = 0;
        }
    }

    return tmpMat;
};

LinearAlgebraHelper.prototype.clone2dMatrix = function (mat) {
    var nr = mat.length;
    var tmpMat = new Array(nr);
    var r = 0;

    for (r = 0; r < nr; r++) {
        tmpMat[r] = mat[r].slice(0);
    }

    return tmpMat;
};

LinearAlgebraHelper.prototype.random2DMatrix = function (nr, nc, epsilon) {
    var e = epsilon || 1;
    var tmpMat = new Array(nr);
    var r = 0;
    var c = 0;

    for (r = 0; r < nr; r++) {
        tmpMat[r] = new Array(nc);
        for (c = 0; c < nc; c++) {
            tmpMat[r][c] = Math.random() * 2 * e - e;
        }
    }

    return tmpMat;
};

LinearAlgebraHelper.prototype.randomVector = function (l, epsilon) {
    var e = epsilon || 1;
    var tmpMat = new Array(l);
    var index;

    for (index = 0; index < l; index++) {
        tmpMat[index] = Math.random() * 2 * e - e;
    }

    return tmpMat;
};

LinearAlgebraHelper.prototype.add2DMatrices = function (mat1, mat2) {
    var nr = mat1.length;
    var nc = mat1[0].length;
    var r = 0;
    var c = 0;
    var tmpMat = new Array(nr);

    for (r = 0; r < nr; r++) {
        tmpMat[r] = new Array(nc);
        for (c = 0; c < nc; c++) {
            tmpMat[r][c] = mat1[r][c] + mat2[r][c];
        }
    }

    return tmpMat;
};

LinearAlgebraHelper.prototype.sub2DMatrices = function (mat1, mat2) {
    var nr = mat1.length;
    var nc = mat1[0].length;
    var r = 0;
    var c = 0;
    var tmpMat = new Array(nr);

    for (r = 0; r < nr; r++) {
        tmpMat[r] = new Array(nc);
        for (c = 0; c < nc; c++) {
            tmpMat[r][c] = mat1[r][c] - mat2[r][c];
        }
    }

    return tmpMat;
};

LinearAlgebraHelper.prototype.mul2DMatrixByScalar = function (mat, s) {
    var nr = mat.length;
    var nc = mat[0].length;
    var r = 0;
    var c = 0;
    var tmpMat = new Array(nr);

    for (r = 0; r < nr; r++) {
        tmpMat[r] = new Array(nc);
        for (c = 0; c < nc; c++) {
            tmpMat[r][c] = mat[r][c] * s;
        }
    }

    return tmpMat;
};

LinearAlgebraHelper.prototype.addBias =  function (mat) {
    var nr = mat.length;
    var r = 0;
    var tmpMat = new Array(nr);

    for (r = 0; r < nr; r++) {
        tmpMat[r] = mat[r].slice(0);
        tmpMat[r].unshift(1);
    }

    return tmpMat;
};

LinearAlgebraHelper.prototype.computeZ = function (Theta, AWB) {
    var nr = AWB.length;
    var r = 0;
    var nc = Theta.length;
    var c = 0;
    var counterLength = AWB[0].length;
    var counter = 0;
    var sum = 0;

    var tmpMat = new Array(nr);

    for (r = 0; r < nr; r++) {
        tmpMat[r] = new Array(nc);
        for (c = 0; c < nc; c++) {
            sum = 0;
            for (counter = 0; counter < counterLength; counter++) {
                sum += AWB[r][counter] * Theta[c][counter];
            }
            tmpMat[r][c] = sum;
        }
    }

    return tmpMat;
};

LinearAlgebraHelper.prototype.computeNextLayer = function (Theta, AWB) {
    var nr = AWB.length;
    var r = 0;
    var nc = Theta.length;
    var c = 0;
    var counterLength = AWB[0].length;
    var counter = 0;
    var sum = 0;

    var tmpMat = new Array(nr);

    for (r = 0; r < nr; r++) {
        tmpMat[r] = new Array(nc);
        for (c = 0; c < nc; c++) {
            sum = 0;
            for (counter = 0; counter < counterLength; counter++) {
                sum += AWB[r][counter] * Theta[c][counter];
            }
            tmpMat[r][c] = 1 / (1 + Math.exp(-1 * sum));
        }
        tmpMat[r].unshift(1);
    }

    return tmpMat;
};

LinearAlgebraHelper.prototype.computeA = function (Z) {
    var nr = Z.length;
    var r = 0;
    var nc = Z[0].length;
    var c = 0;

    var tmpMat = new Array(nr);

    for (r = 0; r < nr; r++) {
        tmpMat[r] = new Array(nc);
        for (c = 0; c < nc; c++) {
            tmpMat[r][c] = 1 / (1 + Math.exp(-1 * Z[r][c]));
        }
    }

    return tmpMat;
};

LinearAlgebraHelper.prototype.sumOfSquares = function (mat) {
    var nr = mat.length;
    var r = 0;
    var nc = mat[0].length;
    var c = 0;

    var sum = 0;

    for (r = 0; r < nr; r++) {
        for (c = 0; c < nc; c++) {
            sum += mat[r][c] * mat[r][c];
        }
    }

    return sum;
};

LinearAlgebraHelper.prototype.setFirstColumnToZeros = function (mat) {
    var nr = mat.length;
    var tmpMat = new Array(nr);
    var r = 0;

    for (r = 0; r < nr; r++) {
        tmpMat[r] = mat[r].slice(0);
        tmpMat[r][0] = 0;
    }

    return tmpMat;
};

LinearAlgebraHelper.prototype.computeD = function (Theta, D, AwithBias) {
    var nr = D.length;
    var nc = Theta[0].length - 1;
    var r = 0;
    var c = 0;
    var sum = 0;
    var counterLength = D[0].length;
    var counter = 0;

    var tmpMat = new Array(nr);

    for (r = 0; r < nr; r++) {
        tmpMat[r] = new Array(nc);
        for (c = 0; c < nc; c++) {
            sum = 0;
            for (counter = 0; counter < counterLength; counter++) {
                sum += Theta[counter][c + 1] * D[r][counter];
            }

            tmpMat[r][c] = sum * AwithBias[r][c + 1] * (1 - AwithBias[r][c + 1]);
        }
    }

    return tmpMat;
};

LinearAlgebraHelper.prototype.computeDTensorSlice = function (AwithBias, D) {
    var nr = D.length;
    var nc = AwithBias.length;
    var r = 0;
    var c = 0;
    var tmpArray = new Array(nr);

    for (r = 0; r < nr; r++) {
        tmpArray[r] = new Array(nc);
        for (c = 0; c < nc; c++) {
            tmpArray[r][c] = AwithBias[c] * D[r];
        }
    }

    return tmpArray;
};

LinearAlgebraHelper.prototype.computeCost = function (a, trainingSetOutput) {
    var nr = a.length;
    var nc = a[0].length;
    var r = 0;
    var c = 0;
    var sum = 0;

    for (r = 0; r < nr; r++) {
        for (c = 0; c < nc; c++) {
            sum += trainingSetOutput[r][c] * Math.log(a[r][c]) + (1 - trainingSetOutput[r][c]) * Math.log(1 - a[r][c]);
        }
    }

    return -1 * sum;
};

LinearAlgebraHelper.prototype.markForDeletion = function (mat) {
    var nr = mat.length;
    var r = 0;

    for (r = 0; r < nr; r++) {
        mat[r] = null;
    }
    mat = null;

}

module.exports = LinearAlgebraHelper;