/**
 *
 * @license
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 */

/**
 * @fileoverview This file defines a function to return an array of the codes associated with each template object of a template.
 */

function generateCodeArrayForTemplate(blocks) {
  var codeList = [];
  var templates = localStorage.getItem("templates");

  if (templates) {
    // template information exists
    templates = JSON.parse(templates);
  } else {
    // no template info exists
    templates = {};
  }

  blocks.forEach((element) => {
    // push code for this element
    codeList.push(templates[element.getFieldValue("name")]["code"]);
  });
  
  return codeList;
}
export default generateCodeArrayForTemplate;
