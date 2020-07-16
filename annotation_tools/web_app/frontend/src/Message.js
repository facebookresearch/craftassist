/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * Message.js implements ASR, send the chat message, switch to the fail or back to settings view
 */

import './Message.css';

import React, { Component } from 'react';
import Button from '@material-ui/core/Button';
import FailIcon from '@material-ui/icons/Cancel';
import IconButton from '@material-ui/core/IconButton';
import List from '@material-ui/core/List';
import ListItem from '@material-ui/core/ListItem';
import ListItemSecondaryAction from '@material-ui/core/ListItemSecondaryAction';
import ListItemText from '@material-ui/core/ListItemText';

const recognition = new window.webkitSpeechRecognition()
recognition.lang = 'en-US'

class Message extends Component {
  constructor(props) {
    super(props);
    this.state = {
      recognizing: false,
      chatResponse: "",
      chats: this.props.chats
    }

    this.toggleListen = this.toggleListen.bind(this)
    this.listen = this.listen.bind(this)
    this.elementRef = React.createRef();
  }

  renderChatHistory() {
    //render the HTML for the chatHistory with a unique key value
    return this.state.chats.map((value, idx) =>
      React.cloneElement(<ListItem>
        <ListItemText
          primary={value.msg}
        />
        <ListItemSecondaryAction>
          {value.msg !== "" ? <IconButton disabled={value.failed} edge="end" aria-label="Fail" onClick={() => this.props.goToQuestion(idx)}>
            <FailIcon />
          </IconButton> : null}
        </ListItemSecondaryAction>
      </ListItem>, {
        key: idx.toString(),
      }),
    );
  }

  isMounted() {
    //check if this element is being displayed on the screen
    return this.elementRef.current != null;
  }

  handleKeyPress(event) {
    //toggle recording if user presses space bar
    if (event.key === " ") {
      this.toggleListen();
      //submit the current message when user presses enter
    } else if (event.key === "Enter") {
      event.preventDefault();
      this.handleSubmit();
    }
  }

  componentDidMount() {
    document.addEventListener("keypress", this.handleKeyPress.bind(this));
  }

  toggleListen() {
    //update the variable and call listen
    this.setState({ recognizing: !this.state.recognizing }, this.listen)
  }


  // we probably don't need this in this version.
  listen() {
    //start listening and grab the output form ASR model to display in textbox
    if (this.state.recognizing) {
      recognition.start()
    } else {
      recognition.stop()
    }
    recognition.onresult = function (event) {
      let msg = ''
      for (var i = 0; i < event.results.length; ++i) {
        if (event.results[i].isFinal) {
          msg += event.results[i][0].transcript;
        }
      }
      document.getElementById('msg').innerHTML = msg;
    }

    recognition.onerror = event => {
      console.log("Error in recognition: " + event.error);
    }
  }

  handleSubmit() {
    console.log("handlesubmit");
    //get the message
    var chatmsg = document.getElementById('msg').innerHTML;
    if (chatmsg.replace(/\s/g, '') !== "") {
      //add to chat history box
      this.state.chats.shift(); // remove first element
      this.state.chats.push({ msg: chatmsg, failed: false }); // add new chat
      this.setState({ chats: this.state.chats });
      //socket connection
      var url = 'http://' + this.props.ipAddress + ':9000/sendchat?message=' + chatmsg + '&ipaddr=' + this.props.ipAddress + '&port=' + this.props.port + '&username=' + this.props.username;

      console.log("fetching from url : " + url);
      fetch(url)
        .then(res => res.text())
        .then(res => this.setState({ chatResponse: res }))
        .then(res => (setTimeout(() => this.setState({ chatResponse: '' }), 2000)))
        .catch(err => err);
      //clear the textbox
      document.getElementById("msg").innerHTML = "";
    }
  }

  render() {
    return (
      <div className="Chat">
        <p>Press spacebar to start/stop recording.</p>
        <p>Click the x next to the message if the outcome wasn't what you intended.</p>
        <List>
          {this.renderChatHistory()}
        </List>
        <div contentEditable="true" className="Msg single-line" id="msg"> </div>
        <Button className="MsgButton" variant="contained" color="primary" onClick={this.handleSubmit.bind(this)} > Submit </Button>
        <Button className="MsgButton" variant="contained" onClick={this.props.goToSettings}> Settings </Button>
        <Button className="SurveyButton" variant="contained" color="primary" onClick={this.props.goToSurvey}> I am done playing </Button>

        <p id="callbackMsg">{this.state.chatResponse}</p>
      </div>
    );
  }
}

export default Message;
