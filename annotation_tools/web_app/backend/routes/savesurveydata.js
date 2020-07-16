/*
 * savesurveydata/<data obj to save> writes the data received in the 
 * query to a json file using fs
 */
var express = require("express");
var router = express.Router();

var fs = require('fs');
var net = require('net');
var path = require('path')

// specify current directory explicitly
var appDir = path.dirname(require.main.filename);
console.log(appDir);
var survey_result_file = path.join(appDir, '../../../../survey_results.json');
console.log("writing survey results to file " + survey_result_file);


router.post("/", function (req, res, next) {
    //handle the post request, save the data to a json file
    var postData = req.body;
    console.log('got the data : ' + postData);

    fs.writeFile(survey_result_file, JSON.stringify(postData), (err) => {
        if (err) throw err;
        console.log('The file has been saved!');
    });

});


module.exports = router;