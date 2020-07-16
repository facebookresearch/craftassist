/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * Settings.js allows users to input the IP Address, port number, and username for connecting to the cuberite server and sending the chat
 */

import React, { Component } from 'react';
import Button from '@material-ui/core/Button';
import FormControl from '@material-ui/core/FormControl';
import TextField from '@material-ui/core/TextField';

class Settings extends Component {
  handleSubmit(event) {
    //check inputs and go to the Message page
    var ip = document.getElementById('formIPAddress').value;
    var port = document.getElementById('formPort').value;
    var user = document.getElementById('formUsername').value;
    this.props.goToMessage(ip, port, user);
  }

  render() {
    return (
      <FormControl noValidate autoComplete="off">
        <TextField
          // inputProps={{
          //   readOnly: true,
          //   disabled: true,
          // }}
          required
          id="formIPAddress"
          label="IP Address"
          defaultValue={this.props.ipAddress}
          margin="normal"
          variant="filled"
        />
        <TextField
          inputProps={{
            readOnly: true,
            disabled: true,
          }}
          required
          id="formPort"
          label="Port"
          defaultValue={this.props.port}
          margin="normal"
          variant="filled"
        />
        <TextField
          required
          id="formUsername"
          label="Username"
          defaultValue={this.props.username}
          margin="normal"
          variant="filled"
        />
        <Button variant="contained" color="primary" onClick={this.handleSubmit.bind(this)}>Submit</Button>
      </FormControl>
    );
  }
}

export default Settings;
