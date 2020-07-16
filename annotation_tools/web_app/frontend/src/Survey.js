/* Copyright (c) Facebook, Inc. and its affiliates. */

import './Survey.css';

import React, { Component } from 'react';
import Button from '@material-ui/core/Button';
import FormControl from '@material-ui/core/FormControl';

class Survey extends Component {
    handleSubmit(event) {
        //check inputs and go to the Message page
        var helpful = document.getElementById('helpful-score').value;
        var helpfulQ = document.getElementById('helpful').textContent;

        var understanding = document.getElementById('bot-understanding-score').value;
        var understandingQ = document.getElementById('bot-understanding').textContent;

        var fun = document.getElementById('fun-score').value;
        var funQ = document.getElementById('fun').textContent;

        var playAgain = document.getElementById('play-again').value;
        var playAgainQ = document.getElementById('play').textContent;

        var playFree = document.getElementById('play-in-free-time').value;
        var playFreeQ = document.getElementById('play-in-free').textContent;

        var recommend = document.getElementById('recommend-bot').value;
        var recommendQ = document.getElementById('recommend').textContent;

        var frustrating = document.getElementById('frustrating-q').value;
        var frustratingQ = document.getElementById('frustrating').textContent;

        var newCommands = document.getElementById('new-commands-q').value;
        var newCommandsQ = document.getElementById('new-commands').textContent;

        var newDialogues = document.getElementById('new-dialogues-q').value;
        var newDialoguesQ = document.getElementById('new-dialogues').textContent;

        var newCapabilities = document.getElementById('new-capabilities-q').value;
        var newCapabilitiesQ = document.getElementById('new-capabilities').textContent;

        var survey_result = {
            newCapabilitiesQ: newCapabilities,
            newDialoguesQ: newDialogues,
            newCommandsQ: newCommands,
            frustratingQ: frustrating,
            recommendQ: recommend,
            playFreeQ: playFree,
            playAgainQ: playAgain,
            funQ: fun,
            understandingQ: understanding,
            helpfulQ: helpful
        }

        var save_survey_data_url = 'http://' + this.props.ipAddress + ':9000/savesurveydata';
        console.log("fetching url: " + save_survey_data_url);
        fetch(save_survey_data_url, {
            method: 'POST',
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(survey_result)
        }).catch(err => console.log(err));

        // Now go to a "thank you" page
        this.props.goToEnd(this.props.ipAddress, this.props.port, this.props.username)
    }


    render() {
        return (
            <FormControl noValidate autoComplete="off" id="form">
                <div><b>Thank you for playing with the bot, please complete this survey so we know how to make the bot better.</b></div>

                <div className="form-group" >
                    <label className="questionLabelsTopmost" id="helpful" >How helpful was the bot overall?</label>
                    <select className="form-control" name="helpful-score" id="helpful-score">
                        <option>1 - Not Very Helpful: The bot made everything harder.</option>
                        <option>2 - Barely Helpful: The bot sometimes made things easier.</option>
                        <option>3 - Helpful: The bot usually made things easier.</option>
                        <option>4 - Very Helpful: The bot made everything easier. </option>
                    </select>
                </div>
                <div class="form-group">
                    <label class="questionLabels" id="bot-understanding">How well did the bot understand what you were saying?</label>
                    <select className="form-control" name="bot-understanding-score" id="bot-understanding-score">
                        <option>1 - Never understood</option>
                        <option>2 - Rarely understood</option>
                        <option>3 - Most of the time understood</option>
                        <option>4 - Always understood</option>
                    </select>
                </div>
                <div class="form-group">
                    <label class="questionLabels" id="fun">How much did you enjoy playing with the bot?</label>
                    <select className="form-control" name="fun-score" id="fun-score">
                        <option>1 - Not At All: Minecraft is more fun without the bot </option>
                        <option>2 - Somewhat: The bot made some activities more fun</option>
                        <option>3 - A Lot: The bot was fun to play with</option>
                        <option>4 - Very Much: The bot made Minecraft so much more fun</option>
                    </select>
                </div>
                <div class="form-group">
                    <label class="questionLabels" id="play">Would you want to play with the bot again ?</label>
                    <select className="form-control" name="play-again" id="play-again">
                        <option>Yes</option>
                        <option>No</option>
                    </select>
                </div>
                <div class="form-group">
                    <label class="questionLabels" id="play-in-free">Would you play with the bot in your free time?</label>
                    <select className="form-control" name="play-in-free-time" id="play-in-free-time">
                        <option>Yes</option>
                        <option>No</option>
                    </select>
                </div>
                <div class="form-group">
                    <label class="questionLabels" id="recommend">Would you recommend the bot to other people?</label>
                    <select className="form-control" name="recommend-bot" id="recommend-bot">
                        <option>Yes</option>
                        <option>No</option>
                    </select>
                </div>
                <div class="form-group">
                    <label class="questionLabels" id="frustrating">What was most frustrating when interacting with the bot? Please be as detailed as possible.</label>
                    <textarea className="form-control" name="frustrating-q" id="frustrating-q"></textarea>
                </div>
                <div class="form-group">
                    <label class="questionLabels" id="new-commands">What new commands would you like for the bot to be able to handle? Please give some examples.</label>
                    <textarea className="form-control" name="new-commands-q" id="new-commands-q"></textarea>
                </div>
                <div class="form-group">
                    <label class="questionLabels" id="new-dialogues">What kinds of dialogues would you like for the bot to be able to handle? Please give some examples.</label>
                    <textarea className="form-control" name="new-dialogues-q" id="new-dialogues-q"></textarea>
                </div>
                <div class="form-group">
                    <label class="questionLabels" id="new-capabilities">What else would you like for the bot to be able to do?</label>
                    <textarea className="form-control" name="new-capabilities-q" id="new-capabilities-q"></textarea>
                </div>
                <Button variant="contained" color="primary" onClick={this.handleSubmit.bind(this)}>Submit</Button>
            </FormControl >

        );
    }
}

export default Survey;
