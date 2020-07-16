/*
 * sendchat/<query> basic socket connection to send data and receive a callback
 */
var express = require("express");
var router = express.Router();
var net = require('net');

router.get("/", function (req, res, next) {
  //send res.query.message to cuberite plugin (ports are hardcoded)
  var client = new net.Socket();

  //parse query, get info from router path parameter (from frontend)
  var chat = req.query.message;
  var portStr = req.query.port;
  var ipaddr = req.query.ipaddr;
  var username = req.query.username;

  if (portStr === "" || ipaddr === "" || username === "" || chat === "") {
    res.send('error, input field(s) are blank');
  } else {
    var port = parseInt(portStr);
    var formattedMessage = username + "::" + chat;
    client.connect(port, ipaddr, function () {
      console.log('Connected');
      client.write(formattedMessage);
    });

    client.on('data', function (data) {
      console.log('Received: ' + data);
      //send callback to frontend to display
      res.send(data);
      client.destroy(); // kill client after server's response
    });

    client.on('close', function () {
      console.log('Connection closed');
    });

    client.on('error', function (ex) {
      console.log("socket connection error");
    });
  }

});

module.exports = router;
