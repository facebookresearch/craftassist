/**
 *
 * @license
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 */

/**
 * @fileoverview This file defines a function to save a template to local storage, and then call upon a function to dump contents of local storage to a file.
 */

import * as Blockly from "blockly/core";
import saveToFile from "../fileHandlers/saveToFile";
import generateCodeArrayForTemplate from "../helperFunctions/generateCodeArrayForTemplate";
import {getSurfaceFormsFromList,generateAllSurfaceFormsForTemplate} from "../helperFunctions/getSurfaceForms";
import getTypes from "../helperFunctions/getTypes";

function saveTemplate(block, name) {
  var allBlocks = Blockly.mainWorkspace.getAllBlocks();
  var types = getTypes(allBlocks);
  console.log(types);
  // string of TO types, space separated
  types = types.join(" ");

  if (localStorage.getItem("templates")) {
    // some templates have been stored already
    var templates = JSON.parse(localStorage.getItem("templates"));
  } else {
    // initialise templates
    templates = {};
  }
  templates[types] = { surfaceForms: "", code: "" };
  templates[types]["surfaceForms"] = generateAllSurfaceFormsForTemplate(allBlocks);
  templates[types]["code"] = generateCodeArrayForTemplate(allBlocks);
  localStorage.setItem("templates", JSON.stringify(templates));
  saveToFile();
}
export default saveTemplate;
