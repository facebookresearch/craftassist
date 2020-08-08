/**
 *
 * @license
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 */

/**
 * @fileoverview This file defines functions to return arrays of surface forms given templates and an array of arrays of surface forms.
 */


/**
 * This is a function to return randomly picked surface forms for each template object within an array of template objects.
 */

function getSurfaceForms(allBlocks) {
  var templates = localStorage.getItem("templates");
  if (templates) {
    templates = JSON.parse(templates);
  }
  var surfaceForms = [];
  allBlocks.forEach((element) => {
    var surfaceForm = templates[element.getFieldValue("name")]["surfaceForms"];
    surfaceForm = randomFromList(surfaceForm);
    surfaceForms.push(surfaceForm);
  });
  return surfaceForms;
}


/**
 * This is a function to return an array of 
  randomly picked surface forms from each element of an array containing lists of surface forms. So, this function takes an array of array of surface forms, and picks out a random element from each element of this array. It returns an array of surface forms.
 */

function getSurfaceFormsFromList(surfaceFormsList) {
  var surfaceForms = [];

  for (let index = 0; index < surfaceFormsList.length; index++) {
    // pick a random surface form from the surface forms at this index
    const surfaceForm = randomFromList(surfaceFormsList[index]);

    // push this surface form
    surfaceForms.push(surfaceForm);
  }

  return surfaceForms;
}


/**
 * This is a function to get all surface forms associated with
 * all elements of an array of template objects. So, this function takes in
 * an array of blocks (template objects) and returns another array
 * containing the surface forms associated with each of them.
 */

function generateAllSurfaceFormsForTemplate(allBlocks) {
  var templates = localStorage.getItem("templates");
  if (templates) {
    // template information exists
    templates = JSON.parse(templates);
  }
  else{
     // no templates saved
     templates={};
  }

  var surfaceForms = [];

  allBlocks.forEach((element) => {
    // get surface form array for this element
    var surfaceForm = templates[element.getFieldValue("name")]["surfaceForms"];
    surfaceForms.push(surfaceForm);
  });
  
  return surfaceForms;
}


// Function that returns a random item from a list
function randomFromList(list) {
    var number_length = list.length;
    // Assemble JavaScript into code variable.
    var x = Math.floor(Math.random() * number_length);
    return list[x];
  }
  
export {getSurfaceForms,getSurfaceFormsFromList,generateAllSurfaceFormsForTemplate}