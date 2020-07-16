package com.example.sttapp;

import android.content.Intent;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;

public class SettingsActivity extends AppCompatActivity {

    EditText ipAddrInput;
    EditText portInput;
    EditText usernameInput;
    Button qrcodeButton;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_settings);

        ipAddrInput = (EditText) findViewById(R.id.ipaddrinput_id);
        portInput = (EditText) findViewById(R.id.portinput_id);
        usernameInput = (EditText) findViewById(R.id.usernameinput_id);
        qrcodeButton = (Button) findViewById(R.id.qrcodebutton_id);

        Intent intent = getIntent();
        ipAddrInput.setText(intent.getStringExtra("ipaddress"));
        portInput.setText(intent.getStringExtra("port"));
        usernameInput.setText(intent.getStringExtra("username"));


        //qrcode button- opens camera activity to detect qr code
        qrcodeButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                goToCameraActivity();
            }

        });
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        /*
         * Show the done/check button on the Action bar
         */
        getMenuInflater().inflate(R.menu.menu_settings, menu);
        return true;
    }

    private void goToCameraActivity() {
        /*
         * create an intent and start it to move to the CameraActivity for QRCode detection
         */
        Intent cameraIntent = new Intent(this, CameraActivity.class);
        startActivityForResult(cameraIntent,1);
    }

    private void goToMainActivity() {
        /*
         * start the intent and
         * send the info back to the main activity (ipaddress, port, username)
         */
        Intent intent = new Intent();
        intent.putExtra("ipaddress",ipAddrInput.getText().toString());
        intent.putExtra("port",portInput.getText().toString());
        intent.putExtra("username",usernameInput.getText().toString());
        setResult(RESULT_OK,intent);
        finish();
    }

    @Override
    public void onBackPressed() {
        /*
         * when the user presses the default back button, send info back to main activity
         */
        goToMainActivity();
        super.onBackPressed();

    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        /*
         * called when a button on the action bar is pressed,
         * handle sending back info to the MainActivity
         */
        switch (item.getItemId()) {
            case R.id.home:
                //go back to MainActivity
                goToMainActivity();
                return true;
            case R.id.check_id:
                goToMainActivity();
        }
        return super.onOptionsItemSelected(item);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        /*
         * when returning to the SettingsActivity from a different Activity
         * handle return from CameraActivity- populate the ip address / port / username
         */
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == 1) { //from the CameraActivity
            if (resultCode == RESULT_OK) {
                ipAddrInput.setText(data.getStringExtra("ipaddress"));
                portInput.setText(data.getStringExtra("port"));
                usernameInput.setText(data.getStringExtra("username"));
            }
        }
    }
}
