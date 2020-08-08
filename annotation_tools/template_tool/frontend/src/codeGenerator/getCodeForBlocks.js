/**
 *
 * @license
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 */

/**
 * @fileoverview This file defines the base code generator function that sets provides corresponding logical-surface forms for templates or template objects in the workspace.
 */

import * as Blockly from "blockly/core";
import getCodeForTemplateObject from "./getCodeForTemplateObject";
import getCodeForTemplate from "./getCodeForTemplate";

function getCodeForBlocks() {
  var topBlock = Blockly.mainWorkspace.getTopBlocks()[0];
  var name = topBlock.getFieldValue("name");
  var childBlocks = topBlock.childBlocks_;
  if (childBlocks.length == 0) {
    // it is a template object

    if (!document.getElementById("surfaceForms").innerText) {
      // If the surface forms box is blank, populate it.
      // use saved information if it is available
      var surfaceFormsOfBlocks = JSON.parse(localStorage.getItem("templates"));

      if (!surfaceFormsOfBlocks[name]) {
        // no surface forms
        window.alert("No saved info, enter surface forms");
        return;
      }
      document.getElementById("surfaceForms").innerText = surfaceFormsOfBlocks[name][
        "surfaceForms"
      ].join("\n");
    }

    var surfaceForms = document
      .getElementById("surfaceForms")
      .innerText.split("\n");

    // to keep track of what goes in generations box
    var textContentBox2 = "";

    // get corresponding (surface form, logical form) for each surface form
    surfaceForms.forEach((surfaceForm) => {
      var code = getCodeForTemplateObject(surfaceForm, topBlock);
      textContentBox2 += code;
    });

    // set the text content of the box which holds generations
    document.getElementById("actionDict").innerText = textContentBox2;
    return textContentBox2;
  } else {
    getCodeForTemplate();
  }
}

export default getCodeForBlocks;
