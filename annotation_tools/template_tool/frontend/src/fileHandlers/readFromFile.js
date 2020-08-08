/**
 *
 * @license
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 */

/**
 * @fileoverview This file contains the definition of a function to restore local storage by requesting the backend for the previously dumped contents of local storage.
 */

import $ from "jquery";

// This function restores local storage information by requesting the backend to provide the dumped contents of local storage.

function restore() {

  const HOST="http://localhost:";
  const PORT="9000";
  fetch(HOST + PORT + "/readAndSaveToFile")
    .then((res) => res.text())
    .then((res) => {
      var res = JSON.parse(res);

      if (res["savedBlocks"]) {
        localStorage.setItem("savedByName", JSON.stringify(res["savedBlocks"]));
      }
      if (res["templates"]) {
        localStorage.setItem("templates", JSON.stringify(res["templates"]));
      }
      if (res["savedByTag"]) {
        localStorage.setItem("tags", JSON.stringify(res["savedByTag"]));
      }
      if (res["spans"]) {
        localStorage.setItem("spans", JSON.stringify(res["spans"]));
        window.spans = res["spans"];
      }
      if (res["blocks"]) {
        localStorage.setItem("blocks", JSON.stringify(res["blocks"]));
      }

      if (!localStorage.getItem("reload")) {
        /* set reload to true and then reload the page */
        localStorage.setItem("reload", "true");
        window.location.reload();
      } else {
        /* after reloading remove "reload" from localStorage */
        localStorage.removeItem("reload");
      }
    });
}

export default restore;
