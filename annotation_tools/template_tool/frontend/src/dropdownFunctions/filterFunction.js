/**
 *
 * @license
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 */

/**
 * @fileoverview This file defines the filter function for the block search dropdown.
 */

function filterFunction() {
  // Declare variables
  var input, filter, listContainer, listElements, txtValue;
  input = document.getElementById("searchInput");
  filter = input.innerText.toUpperCase();
  listContainer = document.getElementById("UL");
  listElements = listContainer.getElementsByTagName("li");

  // Loop through all list items, and hide those who don't match the search query
  for (var i = 0; i < listElements.length; i++) {
    var textContainer= listElements[i].getElementsByTagName("a")[0];
    txtValue = textContainer.textContent || textContainer.innerText;
    if (txtValue.toUpperCase().indexOf(filter) > -1) {
      listElements[i].style.display = "";
    } else {
      listElements[i].style.display = "none";
    }
  }
}

export default filterFunction;
