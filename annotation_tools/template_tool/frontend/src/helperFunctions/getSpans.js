/**
 *
 * @license
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 */

/**
 * @fileoverview This file defines a function to return the spans 
 * associated with surface forms
 */

function getSpans(surfaceForms) {
    console.log(surfaceForms);
    var spans = [];
    for (var i = 0; i < surfaceForms.length; i++) {
      var surfaceForm = surfaceForms[i];
      var span = window.spans[surfaceForm];
      if (!span) {
        // the entire surface form is the span
        span = surfaceForm;
      }
      spans.push(span);
    }
    console.log(spans);
    return spans;
  }
  export default getSpans