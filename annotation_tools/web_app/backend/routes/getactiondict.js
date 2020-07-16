/*
 * getactiondict/<message> basic socket connection to receive the corresponding action dictionary to the message
 * starts a server side socket that listens for incoming action dictionary
 */
var express = require("express");
var router = express.Router();
var net = require('net');
var port = 2557; //hardcoded port number
var ipaddr = "127.0.0.1" //hardcoded ip address

router.get("/", function (req, res, next) {
  //handle get request from frontend
  //parse query, get info from router path parameter (from frontend)
  console.log("okokokok");
  //send message to get callback action dict
  var client = new net.Socket();

  var chat = req.query.message;

  client.connect(port, ipaddr, function () {
    console.log('Connected');
    client.write(chat);
  });

  client.on('data', function (data) {
    console.log('Received: ' + data);
    // this needs to be done only when sending over network (from cluster)
    var dict_str = decodeURIComponent(escape(data));
    var data2 = JSON.parse(dict_str);
    res.send(data2);
    client.destroy(); // kill client after server's response
  });

  client.on('close', function () {
    console.log('Connection closed');
  });

  client.on('error', function (ex) {
    console.log("socket connection error");
  });

});

module.exports = router;
