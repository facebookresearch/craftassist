/*
 * MainActivity connects the front end xml, implements asr, and processes the user input.
 */

package com.example.sttapp;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.speech.RecognitionListener;
import android.speech.RecognizerIntent;
import android.speech.SpeechRecognizer;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;

import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Toast;

import java.util.ArrayList;

public class MainActivity extends AppCompatActivity implements RecognitionListener {
    EditText textInput;
    Button submitButton;
    Button speechButton;

    String ipAddress = "";
    String portStr = "";
    String username = "";

    SpeechRecognizer speechRecognizer;
    Intent recognizerIntent;
    final int REQUEST_CODE_RECORD_AUDIO = 100;
    boolean recording = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        /*
         * connect the UI components from activity_main.xml,
         * initialize the speech recognizer for ASR and handle response when the speech button is pressed
         * handle the response when the QR code button is pressed (go to CameraActivity)
         */
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        textInput = (EditText) findViewById(R.id.textinput_id);
        submitButton = (Button) findViewById(R.id.submitbutton_id);
        speechButton = (Button) findViewById(R.id.speechbutton_id);

        //initialize speech recognizer
        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this);
        Log.d("MYSPEECHRECOGNIZER","recognition available: "+SpeechRecognizer.isRecognitionAvailable(this));
        speechRecognizer.setRecognitionListener(this);
        recognizerIntent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
        recognizerIntent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_PREFERENCE,"en"); //set english as the language
        recognizerIntent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL,RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
        recognizerIntent.putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 3);

        //called when the "record" button is pressed - toggles the speech recognizer on/off
        speechButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (recording) {
                    stopRecording();
                } else {
                    //check permission for recording audio
                    if (ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
                        //permission not granted so request permissions
                        ActivityCompat.requestPermissions(MainActivity.this, new String[]{Manifest.permission.RECORD_AUDIO}, REQUEST_CODE_RECORD_AUDIO);
                    } else {
                        //permission already granted
                        startRecording();
                    }
                }
            }
        });

        //when button is clicked, read the input, and create an instance of Client accordingly.
        submitButton.setOnClickListener(
        new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                String message = textInput.getText().toString();
                String toastMessage = "";
                int port = 0;

                // for testing: set defaults if ip/port not provided later, handle empty error check
                if (ipAddress.isEmpty()) {
                    toastMessage = "Please specify IP address";
                }
                if (portStr.isEmpty()) {
                    toastMessage = "Please specify port";
                } else {
                    port = Integer.parseInt(portStr);
                }
                if (message.isEmpty()) {
                    toastMessage = "Cannot send empty message";
                }
                if (username.isEmpty()) {
                    toastMessage = "Please specify the username of player";
                }

                // TODO: can have a toast indicating that message was received?
                if (!toastMessage.isEmpty()) {
                    Toast toast = Toast.makeText(getBaseContext(), toastMessage, Toast.LENGTH_SHORT);
                    toast.show();
                } else {
                    Client client = new Client(MainActivity.this, ipAddress, port, username, message);
                    client.execute();
                    textInput.getText().clear();
                }
            }
        });
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        /*
         * Show the settings button on the Action bar
         */
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }


    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        /*
         * called when a button on the action bar is pressed,
         * opens the settings page when pressed
         */
        switch (item.getItemId()) {
            case R.id.settings_id:
                //show settings page
                Intent intent = new Intent(this, SettingsActivity.class);
                intent.putExtra("ipaddress",ipAddress);
                intent.putExtra("port",portStr);
                intent.putExtra("username",username);
                startActivityForResult(intent,1);
        }
        return super.onOptionsItemSelected(item);
    }


    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        /*
         * when returning to the MainActivity from a different Activity
         * handle return from SettingsActivity- populate the ip address / port / username
         */
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == 1) { //from the SettingsActivity
            if (resultCode == RESULT_OK) {
                ipAddress = "" + data.getStringExtra("ipaddress");
                portStr = "" + data.getStringExtra("port");
                username = "" + data.getStringExtra("username");
            }
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        /*
         * After user has granted permissions, start recording, else do nothing since permission not granted (show toast)
         */
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        switch (requestCode) {
            case REQUEST_CODE_RECORD_AUDIO:
                if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    Log.d("MYSPEECHRECOGNIZER","record audio permission granted");
                    startRecording();
                } else {
                    Log.d("MYSPEECHRECOGNIZER","record audio permission denied");
                    Toast toast = Toast.makeText(getBaseContext(), "permission denied, cannot record audio",Toast.LENGTH_SHORT);
                    toast.show();
                }

        }
    }

    //recognition listener methods:

    @Override
    public void onReadyForSpeech(Bundle bundle) { }

    @Override
    public void onBeginningOfSpeech() { }

    @Override
    public void onRmsChanged(float v) { }

    @Override
    public void onBufferReceived(byte[] bytes) { }

    @Override
    public void onEndOfSpeech() {
        /*
         * if no more speech audio is detected then stop recording
         */
        stopRecording();
    }

    @Override
    public void onError(int i) {
        /*
         * stop recording when an error occurs
         */
        Log.d("MYSPEECHRECOGNIZER","error");
        if (recording) {
            stopRecording();
        }
        Toast toast = Toast.makeText(getBaseContext(), "could not understand audio input", Toast.LENGTH_SHORT);
        toast.show();
    }

    @Override
    public void onResults(Bundle bundle) {
        /*
         * use the text with the most confidence and populate in the message field
         */
        ArrayList<String> results = bundle.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION);
        Log.d("MYSPEECHRECOGNIZER","results: "+results.toString());
        String bestResult = results.get(0);
        if (recording) {
            stopRecording();
        }
        textInput.setText(bestResult);
    }

    @Override
    public void onPartialResults(Bundle bundle) { }

    @Override
    public void onEvent(int i, Bundle bundle) { }

    private void startRecording() {
        /*
         * set boolean to indicate recording and start the speechRecognizer.
         * change the button UI to indicate recording
         */
        recording = true;
        speechRecognizer.startListening(recognizerIntent);
        speechButton.setText("recording");
        speechButton.setBackgroundColor(Color.GRAY);
    }

    private void stopRecording() {
        /*
         * set boolean to indicate not recording and stop the speechRecognizer.
         * change the button UI to indicate not recording
         */
        recording = false;
        speechRecognizer.stopListening();
        speechButton.setText("record");
        speechButton.setBackgroundColor(Color.LTGRAY);
    }
}
