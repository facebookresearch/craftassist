/**
 *
 * @license
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 */

/**
 * @fileoverview This file defines a function to save a template object to local storage, and then call upon a function to dump contents of local storage to a file.
 */

import saveToFile from "../fileHandlers/saveToFile";

// This function saves a template object to local storage, and then calls upon the backend to save it to a file

function saveTemplateObject(block, name) {
  var surfaceForms = document
    .getElementById("surfaceForms")
    .innerText.split("\n");

  var actionDict = document.getElementById("actionDict").innerText;
  if (actionDict) {
    // associated with code
    actionDict = JSON.parse(actionDict);
  } else {
    actionDict = undefined;
  }
  var templateObjectsSaved = localStorage.getItem("templates");
  if (templateObjectsSaved) {
    templateObjectsSaved = JSON.parse(templateObjectsSaved);
  } else {
    templateObjectsSaved = {};
  }

  templateObjectsSaved[name] = {};
  templateObjectsSaved[name]["surfaceForms"] = surfaceForms;
  templateObjectsSaved[name]["code"] = actionDict;
  console.log(templateObjectsSaved);
  localStorage.setItem("templates", JSON.stringify(templateObjectsSaved));

  saveToFile(); // dump new local storage information to file
}

export default saveTemplateObject;
