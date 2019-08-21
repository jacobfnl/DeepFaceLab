const webpack = require('webpack');
const config = {
        devtool: 'eval-source-map',
 entry: __dirname + '/js/index.jsx',
 output:{
  path: __dirname + '/../public',
  filename: 'bundle.js',
},
 resolve: {
  extensions: ['.js','.jsx','.css']
 },
 module: {
  rules: [
  {
   test: /\.jsx?/,
   loader: 'babel-loader',
   exclude: /node_modules/,
   query:{
    presets: ["@babel/preset-react"]
  }
  },
  {
         test: /\.css$/,
         loader: 'style-loader!css-loader?modules'
  }]
}
};
module.exports = config;