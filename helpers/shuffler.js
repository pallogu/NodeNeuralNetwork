var _ = require('lodash');
var knuthShuffle = require('knuth-shuffle').knuthShuffle;

var Shuffler = function (){};

_.extend(Shuffler.prototype, {
    shuffle: function (array) {
        'use strict';
        knuthShuffle(array);
        return array;
    },
    reshuffle: function (Xmatrix, Ymatrix) {
        'use strict';

        var tmpArray = [];
        var tmpX = [];
        var tmpY = [];
        var i;

        if(Xmatrix.length !== Ymatrix.length) {
            throw 'Shuffler: reshuffle method: Length of arrays do not match';
        } else {
            for(i = 0; i < Xmatrix.length; i = i + 1) {
                tmpArray.push([Xmatrix[i], Ymatrix[i]]);
            }

            knuthShuffle(tmpArray);

            for(i = 0; i < Xmatrix.length; i = i + 1) {
                tmpX.push(tmpArray[i][0]);
                tmpY.push(tmpArray[i][1]);
            }
        }

        return [tmpX, tmpY];
    }
});


module.exports = new Shuffler();