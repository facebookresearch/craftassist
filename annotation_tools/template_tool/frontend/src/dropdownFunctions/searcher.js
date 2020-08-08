/**
 *
 * @license
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 */

/**
 * @fileoverview This file defines a search function that places the block searched for by the user in the dropdown into the toolbox.
 */

import Blockly, { Generator } from "blockly/core";
import $ from "jquery";
function searchForBlocks() {
  // add a default block to toolbox
  document.getElementById(
    "toolBox"
  ).innerHTML = `<block xmlns="https://developers.google.com/blockly/xml" type="customBlock"><field name="optional">TRUE</field><field name="text">Customisable block</field></block>`;
  Blockly.mainWorkspace.updateToolbox(document.getElementById("toolBox"));

  // name/tag the user wants
  var nameOrTag = document.getElementById("searchInput").innerText;

  var taggedInfoString = localStorage.getItem("tags");
  var taggedInfo;

  if (taggedInfoString) {
    // search in tags
    taggedInfo = JSON.parse(taggedInfoString);
    if (taggedInfo[nameOrTag]) {
      // this tag exists
      var blocks = taggedInfo[nameOrTag];
      blocks.forEach((element) => {
        var blockDom = Blockly.Xml.textToDom(element);
        console.log(blockDom);
        var blockInfo = blockDom.firstChild;

        // append the block to the toolbox and update it
        $("#toolBox").append(blockInfo);
        Blockly.mainWorkspace.updateToolbox(document.getElementById("toolBox"));
      });
    }
  }
  var namedInfoString= localStorage.getItem("savedByName");
  var namedInfo;
  if (namedInfoString) {
    // search in names
    namedInfo = JSON.parse(namedInfoString);
    if (namedInfo[nameOrTag]) {
      // this name exists
      var block = namedInfo[nameOrTag];
      var blockDom = Blockly.Xml.textToDom(block);
      var blockInfo = blockDom.firstChild;
      $("#toolBox").append(blockInfo);
      Blockly.mainWorkspace.updateToolbox(document.getElementById("toolBox"));
    }
  }
}

export default searchForBlocks;
