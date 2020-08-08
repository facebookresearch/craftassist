import * as Blockly from "blockly/core";
import saveToFile from "../../fileHandlers/saveToFile";
import $ from "jquery";

const tagBlockCallback = (block) => {
  console.log(block);
  var blockAsText = Blockly.Xml.domToText(Blockly.Xml.blockToDom(block, true));
  var fullBlockXML = `<xml xmlns="https://developers.google.com/blockly/xml">${blockAsText}</xml>`;
  var tag = window.prompt("Please enter a tag.");
  var currentTagInfo = localStorage.getItem("tags");
  if (currentTagInfo) {
    currentTagInfo = JSON.parse(currentTagInfo);
  } else {
    currentTagInfo = {};
  }
  var infoOfCurTag = currentTagInfo[tag];
  if (!infoOfCurTag) {
    infoOfCurTag = [];
  }
  infoOfCurTag.push(fullBlockXML);
  currentTagInfo[tag] = infoOfCurTag;
  localStorage.setItem("tags", JSON.stringify(currentTagInfo));
  var currentDropdownInfo = JSON.parse(localStorage.getItem("blocks"));
  currentDropdownInfo.push(tag);
  localStorage.setItem("blocks", JSON.stringify(currentDropdownInfo));

  saveToFile();
  window.location.reload(true);
};

export default tagBlockCallback;
