/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * App.js handles displaying/switching between different views (settings, message, and TODO: fail)
 */
import './App.css';
import React, { Component } from 'react';
import Message from './Message';
import Question from './Question';
import Settings from './Settings';
import Survey from './Survey';
import ThankYou from './ThankYou';


class App extends Component {
  constructor(props) {
    super(props);
    var url_parts = window.location.href;
    var params = this.getUrlParameterByName('ip_addr'); // get value of ip address from url
    this.state = {
      currentView: 0,
      ipAddress: params,
      port: 2556,
      username: "",
      chats: [{ msg: "", failed: false }, { msg: "", failed: false }, { msg: "", failed: false }, { msg: "", failed: false }, { msg: "", failed: false }],
      failidx: -1
    }
    this.MessageRef = React.createRef();
  }


  getUrlParameterByName(name) {
    var match = RegExp('[?&]' + name + '=([^&]*)').exec(window.location.search);
    return match && decodeURIComponent(match[1].replace(/\+/g, ' '));
  }

  goToSettings() {
    //change the state to switch the view to show Settings
    const currMessageRef = this.MessageRef.current;
    this.setState({ currentView: 0, chats: currMessageRef.state.chats });
  }

  goToSurvey() {
    // change the state to show the survey page
    const currMessageRef = this.MessageRef.current;
    this.setState({ currentView: 3, chats: currMessageRef.state.chats });
  }

  goToEnd() {
    // change the state to show the survey page
    this.setState({ currentView: 4 });
  }

  // something happens here if we come back after submitting.
  goToMessage(i, p, u) {
    //change the state to switch the view to show Message and save the user input necessary for socket connection
    var newchats = this.state.chats;
    if (this.state.failidx !== -1) {
      newchats[this.state.failidx].failed = true;
    }
    this.setState({ currentView: 1, ipAddress: i, port: p, username: u, chats: newchats });
  }

  goToQuestion(idx) {
    //change the state to switch view to show Fail page
    const currMessageRef = this.MessageRef.current;
    this.setState({ currentView: 2, chats: currMessageRef.state.chats, failidx: idx });
  }
  render() {
    return (
      <div className="App">
        <header className="App-header">
          Minecraft Web
        </header>
        <div>

        </div>
        <div className="content">
          {this.state.currentView === 0 ? <Settings goToMessage={this.goToMessage.bind(this)} ipAddress={this.state.ipAddress} port={this.state.port} username={this.state.username} /> : null}
          {this.state.currentView === 1 ? <Message ref={this.MessageRef} goToSurvey={this.goToSurvey.bind(this)} goToSettings={this.goToSettings.bind(this)} goToQuestion={this.goToQuestion.bind(this)} ipAddress={this.state.ipAddress} port={this.state.port} username={this.state.username} chats={this.state.chats} /> : null}
          {this.state.currentView === 2 ? <Question chats={this.state.chats} failidx={this.state.failidx} goToMessage={this.goToMessage.bind(this)} ipAddress={this.state.ipAddress} port={this.state.port} username={this.state.username} failmsg={this.state.chats[this.state.failidx].msg} /> : null}
          {this.state.currentView === 3 ? <Survey goToEnd={this.goToEnd.bind(this)} ipAddress={this.state.ipAddress} port={this.state.port} username={this.state.username} /> : null}
          {this.state.currentView === 4 ? <ThankYou ipAddress={this.state.ipAddress} port={this.state.port} username={this.state.username} /> : null}

        </div>
      </div>
    );
  }

}

export default App;