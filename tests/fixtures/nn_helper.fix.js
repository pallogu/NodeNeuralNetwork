

var g = function (z) {
    return 1 / ( 1  + Math.exp(-1 * z));
};

var gPerElement = function (matrix) {

    var processedMatrix = [];

    for(var k = 0; k < matrix.length; k++ ) {
        processedMatrix.push(matrix[k].slice());
    }

    var r = processedMatrix.length;
    var c = processedMatrix[0].length;

    for(var i = 0; i < r; i++) {
        for (var j = 0; j < c; j++) {
            processedMatrix[i][j] = g(processedMatrix[i][j]);
        }
    }

    return processedMatrix;
};

var addBias = function (matrix) {
    var matrixWithBias = matrix.slice();
    return matrixWithBias.map(function (example) {
        var newExample = example.slice();
        newExample.unshift(1);
        return newExample;
    });
};

var computeZ = function (aWithBias, theta) {
    var newExamples = [];

    for (var example = 0; example <  aWithBias.length; example++) {
        //this is loop through examples
        var newExample = [];
        for (r = 0; r < theta.length; r++) {
            var sum = 0;
            for (var c = 0; c < theta[0].length; c++) {
                sum += theta[r][c] * aWithBias[example][c];
            }
            newExample.push(sum);
        }
        newExamples.push(newExample);
    }
    return newExamples;
};

var sumThetaSquared = function (mat) {
    var sum = 0;

    for(var row = 0; row < mat.length; row++) {
        for (var column = 0; column < mat[row].length; column++) {
            sum += mat[row][column] * mat[row][column];
        }
    }

    return sum;
}

var trainingSetInput = [
    [0,0],
    [0,1],
    [1,0],
    [1,1]
];

var a0  = trainingSetInput.slice(0);

var trainingSetOutput = [
    [0],
    [1],
    [1],
    [0]
];

var Theta1 = [[1, 1.1, 1.2], [2, 2.1, 2.2]];
var Theta2 = [[1, 1.1, 1.2], [2, 2.1, 2.2]];
var Theta3  = [[1,1.1,1.2]];

var lamda = 0.1;

var a0WithBias = addBias(a0);

var z1 = computeZ(a0WithBias, Theta1);
var a1 = gPerElement(z1);
var a1WithBias = addBias(a1);

var z2 = computeZ(a1WithBias, Theta2);
var a2 = gPerElement(z2);
var a2WithBias = addBias(a2);

var z3 = computeZ(a2WithBias, Theta3);
var a3 = gPerElement(z3);

var cost = 0;

for(var exampleIndex = 0 ; exampleIndex < a3.length; exampleIndex++) {

    for (var predictionIndex = 0; predictionIndex < a3[exampleIndex].length; predictionIndex++) {
        cost += trainingSetOutput[exampleIndex][predictionIndex] * Math.log(a3[exampleIndex][predictionIndex]);
        cost += (1-trainingSetOutput[exampleIndex][predictionIndex]) * Math.log(1 - a3[exampleIndex][predictionIndex]);
    }
}
cost = -1 * cost;
cost += sumThetaSquared(Theta1)*lamda/2;
cost += sumThetaSquared(Theta2)*lamda/2;
cost += sumThetaSquared(Theta3)*lamda/2

var d3 = (function (){
    var tmpArray = [];
    for (var i = 0; i < a3.length; i++) {
        var elementArray = [];

        for (var j = 0; j < a3[i].length; j++) {
            var d = a3[i][j] - trainingSetOutput[i][j]
            elementArray.push(1*d.toPrecision(16));
        }
        tmpArray.push(elementArray);
    }
    return tmpArray;
}());

module.exports = {
    Theta1 : Theta1,
    Theta2 : Theta2,
    Theta3 : Theta3,
    trainingSetInput : trainingSetInput,
    trainingSetOutput : trainingSetOutput,
    a0WithBias : a0WithBias,
    z1 : z1,
    a1 : a1,
    a1WithBias : a1WithBias,
    z2 : z2,
    a2 : a2,
    a2WithBias : a2WithBias,
    z3 : z3,
    a3 : a3,
    d3 : d3,
    cost: cost,
    lamda: lamda
}