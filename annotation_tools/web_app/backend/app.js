var createError = require('http-errors');
var express = require('express');
var path = require('path');
var logger = require('morgan');
var cors = require("cors");

//define route for handling the path
var sendchatRouter = require('./routes/sendchat');
var getactiondictRouter = require('./routes/getactiondict');
var savedataRouter = require('./routes/savedata');
var savesurveydataRouter = require('./routes/savesurveydata');

var app = express();

// view engine setup
app.set('views', path.join(__dirname, 'views'));
app.set('view engine', 'jade');
app.use(cors());

app.use(logger('dev'));
app.use(express.json());
app.use(express.urlencoded({ extended: false }));

//add route for handling the path sendchat, getactiondict
app.use('/sendchat', sendchatRouter);
app.use('/getactiondict', getactiondictRouter);
app.use('/savedata', savedataRouter);
app.use('/savesurveydata', savesurveydataRouter);

// catch 404 and forward to error handler
app.use(function (req, res, next) {
  next(createError(404));
});

// error handler
app.use(function (err, req, res, next) {
  // set locals, only providing error in development
  res.locals.message = err.message;
  res.locals.error = req.app.get('env') === 'development' ? err : {};

  // render the error page
  res.status(err.status || 500);
  res.render('error');
});

module.exports = app;