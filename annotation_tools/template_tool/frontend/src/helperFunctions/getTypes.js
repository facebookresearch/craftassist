/**
 *
 * @license
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 */

/**
 * @fileoverview This file defines a function to return an array of the types associated with a list of template objects.
 */

function getTypes(blocks) {
  var typeList = [];

  blocks.forEach((element) => {
    // push the type/name of this template object
    typeList.push(element.getFieldValue("name"));
  });

  return typeList;
}

export default getTypes;
