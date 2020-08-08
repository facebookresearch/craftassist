/**
 *
 * @license
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 */

/**
 * @fileoverview This file contains the callback function for the "saveBlock" option that the custom block has. This saves information about the block to local storage and the template library file.
 */

import $ from "jquery";
import saveTemplateObject from "../../saveToLocalStorage/saveTemplateObject";
import * as Blockly from "blockly/core";
import saveTemplate from "../../saveToLocalStorage/saveTemplate";

function saveBlockCallback(block) {
  var blockAsText = Blockly.Xml.domToText(Blockly.Xml.blockToDom(block, true));

  // wrap the block in xml tags
  var fullBlockXML = `<xml xmlns="https://developers.google.com/blockly/xml">${blockAsText}</xml>`;

  var name = block.getFieldValue("name");
  var allBlocks = Blockly.mainWorkspace.getAllBlocks();
  if (allBlocks.length != 1) {
    // not a single template object
    name = window.prompt("Enter a name for the template");
  }

  // get the blocks currently saved by name
  var currentSavedInfoString = localStorage.getItem("savedByName");
  var currentSavedInfo;

  if (currentSavedInfoString) {
    // blocks have already been saved
    currentSavedInfo = JSON.parse(currentSavedInfoString);
  } else {
    // no blocks saved, initialise the dictionary.
    currentSavedInfo = {};
  }

  // save this block
  currentSavedInfo[name] = fullBlockXML;

  localStorage.setItem("savedByName", JSON.stringify(currentSavedInfo));

  var currentDropdownInfo = JSON.parse(localStorage.getItem("blocks"));
  currentDropdownInfo.push(name);
  localStorage.setItem("blocks", JSON.stringify(currentDropdownInfo));

  // wrap the block name in option tags
  if (allBlocks.length == 1) {
    // it is a template object
    saveTemplateObject(block, name);
  } else {
    saveTemplate(block, name);
  }

  // refresh the dropdown selections
  window.location.reload(true);
}

export default saveBlockCallback;
