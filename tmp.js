var M = require('eigenjs').Matrix
    , mat = M.Random(2, 3);
var t = JSON.parse(M.Random(1, 37).toString({matPrefix: '[', matSuffix: ']', dontAlingColumnts:true, coeffSeparator: ',', colSepartor: ''}));
console.log('tmp', t);