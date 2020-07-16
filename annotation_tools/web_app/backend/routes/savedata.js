/*
 * savedata/<data obj to save> writes the data received in the query to a database using sqlite3
 */
var express = require("express");
var router = express.Router();
var net = require('net');

var path = require('path')
// // specify current directory explicitly
var appDir = path.dirname(require.main.filename);
console.log(appDir);
var db_file_name = path.join(appDir, '../../../../web_app_data.db');
console.log(db_file_name);


const sqlite3 = require('sqlite3').verbose();

//set up database
let db = new sqlite3.Database(db_file_name, (err) => {
  if (err) {
    //cannot open database
    console.error(err.message);
  } else {
    //try to create a new table

    // old DB schema with adtt and labeling tool in place
    /* db.run(`CREATE TABLE data (
      error_type TEXT,
      chat TEXT,
      adtt TEXT,
      old_dict TEXT,
      updated_dict TEXT,
      feedback TEXT
    )`, (err) => {
      if (err) {
        console.log('table exists')
      } else {
        console.log('created new table')
      }
    }); */

    // new updated DB schema
    db.run(`CREATE TABLE data (
      error_type TEXT,
      chat TEXT,
      action_dict TEXT,
      feedback TEXT
    )`, (err) => {
      if (err) {
        console.log('table exists');
      } else {
        console.log('created new table called : data in ' + db_file_name);
      }
    });
  }
});


db.close();

router.post("/", function (req, res, next) {
  //handle the post request, save the data to database
  var postData = req.body;

  let db = new sqlite3.Database(db_file_name, (err) => {
    if (err) {
      //cannot open database
      console.error(err.message);
    } else {
      //parse data
      var action_dict = JSON.stringify(postData.action_dict);
      // var new_dict = JSON.stringify(postData.new_action_dict);
      var err_type = "NOT_IDENTIFIED"
      /* if (postData.asr) {
        err_type = "ASR";
      } else if (postData.adtt) {
        err_type = "ADTT"
      } */
      if (postData.parsing_error) {
        err_type = "PARSING_ERROR";
      }
      var msg = postData.msg;
      var feedback = postData.feedback;
      // var adtt = postData.adtt_text;

      //save to database
      /* old insert
      db.run('INSERT INTO data (error_type, chat, adtt, old_dict, updated_dict, feedback) VALUES (?,?,?,?,?,?)', [err_type, msg, adtt, old_dict, new_dict, feedback]);
      */
      db.run('INSERT INTO data (error_type, chat, action_dict, feedback) VALUES (?,?,?,?)', [err_type, msg, action_dict, feedback], (err) => {
        if (err) {
          console.log('cannot insert into table :' + db_file_name);
        } else {
          console.log('written user feedback to :' + db_file_name);
        }
      });

    }
  });
  db.close();
});

module.exports = router;
