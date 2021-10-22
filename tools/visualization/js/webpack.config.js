const path = require('path');

module.exports = {
    entry: './src/index.js',
    devtool: 'eval-cheap-source-map',
    output: {
        filename: 'main.js',
        path: path.resolve(__dirname, 'dist'),
    },
};