/**
 *
 * @license
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 */

/**
 * @fileoverview This file defines the main Blockly component and basic layout of the template generator.
 */

import React from "react";
import "./App.css";
import "blockly/blocks";
import "./block/customBlock";
import restore from "./fileHandlers/readFromFile";
import Button from "@material-ui/core/Button";
import Select from "@material-ui/core/Select";
import $ from "jquery";
import filterFunction from "./dropdownFunctions/filterFunction";
import searchForBlocks from "./dropdownFunctions/searcher";
import highlightSelectedText from "./highlightSelectedText";
import getCodeForBlocks from "./codeGenerator/getCodeForBlocks";
import BlocklyComponent, { Block, Value, Field, Shadow } from "./Blockly";

import BlocklyJS from "blockly/javascript";

var spans = localStorage.getItem("spans");
if (spans) {
  // saved information about spans exists
  window.spans = JSON.parse(spans);
} else {
  // no saved information, initialise
  window.spans = {};
}

class App extends React.Component {
  constructor(props) {
    super(props);
    this.simpleWorkspace = React.createRef();
  }
  componentDidMount() {
    restore();
  }

  generateCode = () => {
    // clear the boxes to hold generations
    clear();
    var numberOfGenerations = document.getElementById("numberOfGen").value;
    if (!numberOfGenerations){
      // no input has been provided, default to 1 generation
      numberOfGenerations = 1;
    }
    var i = 0;
    while (i < numberOfGenerations) {
      i++;
      getCodeForBlocks();
    }
  };

  render() {
    return (
      <div className="App">
        <header className="App-header">
          <p> Blockly based template generator</p>
          <BlocklyComponent
            ref={this.simpleWorkspace}
            readOnly={false}
            trashcan={true}
            media={"media/"}
            move={{
              scrollbars: true,
              drag: true,
              wheel: true,
            }}
          >
            <Block type="customBlock" />
          </BlocklyComponent>
        </header>

        <div id="logicalAndSurfaceForms">
          <h4 id="dropdownHeading">Blocks</h4>

          <h4 id="heading1"> surface forms</h4>
          <div id="surfaceForms" contentEditable="true"></div>
          <h4 id="heading2"> Action dictionary-surface forms</h4>
          <pre id="actionDict" contentEditable="true"></pre>

          <div
            id="searchInput"
            contentEditable="true"
            onKeyUp={filterFunction}
            placeholder="Search for blocks.."
          ></div>
          <ul id="UL">{listItems}</ul>
          <Button
            id="highlight"
            variant="contained"
            color="primary"
            onClick={highlightSelectedText}
          >
            Highlight
          </Button>
          <input type="color" id="colors"></input>
          <Button
            id="generator"
            variant="contained"
            color="primary"
            onClick={this.generateCode}
          >
            Generate code
          </Button>

          <input id="numberOfGen" placeholder="Number of generations"></input>

          <Button
            id="clear"
            variant="contained"
            color="primary"
            onClick={clear}
          >
            Clear boxes
          </Button>
        </div>
      </div>
    );
  }
}

export default App;

var textList = localStorage.getItem("blocks");
var text;
if (textList) {
  // the dropdown has been populated
  text = JSON.parse(textList);
} else {
  // the dropdown has only the default element
  text = ["Custom block"];
  localStorage.setItem("blocks", JSON.stringify(text));
}
var listItems = text.map((str) => (
  // map each string to a list element
  <li>
    <a
      onClick={() => {
        document.getElementById("searchInput").innerText = str;
        searchForBlocks();
      }}
    >
      {str}
    </a>
  </li>
));

// This function clears the data holding boxes
function clear() {
  document.getElementById("surfaceForms").innerText = "";
  document.getElementById("actionDict").innerText = "";
}
