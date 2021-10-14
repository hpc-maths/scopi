const express = require('express');
const app = express();
const path = require('path');
const favicon = require('serve-favicon')
const fs = require('fs');

app.use(express.static(__dirname + '/public'))
app.use('/build/', express.static(path.join(__dirname, 'node_modules/three/build')));
app.use('/dat/', express.static(path.join(__dirname, 'node_modules/dat.gui/build')));
app.use('/jsm/', express.static(path.join(__dirname, 'node_modules/three/examples/jsm')));
app.use(favicon(path.join(__dirname, 'public', 'favicon.ico')));
console.log(path.join(__dirname, 'public', 'favicon.ico'));

// exemple : http://localhost:8080/api/json?date=2020-07-15
app.get('/api/json', (req, res) => {
  // console.log(req);
  console.log('Load file : ' + '../../../build/' + req.query.filename);
  fs.readFile('../../../build/' + req.query.filename, (err, json) => {
    let obj = JSON.parse(json);
    res.json(obj);
  });
});

app.listen(8080, () => {
  console.log('Visit http://localhost:8080');
});
