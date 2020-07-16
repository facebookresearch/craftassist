/*
 * Client class that tries to connect to the specified address and port.
 * if connection is successful, the provided message will be sent over with specified username.
 * listens for a callback message (indicating success/error) on server side.
 */
package com.example.sttapp;

import android.app.Activity;
import android.content.Context;
import android.os.AsyncTask;
import android.util.Log;
import android.widget.Toast;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.net.Socket;

public class Client extends AsyncTask<Void, Void, Void> {
    String addr;
    int port;
    String message = "";
    String username;
    Context context;
    Activity activity;

    public Client(Activity c, String a, int p, String u, String s) {
        addr = a;
        port = p;
        message = s;
        username = u;
        activity = c;
    }

    @Override
    protected Void doInBackground(Void... voids) {
        /*
         * creates a socket with ip addr and port, formats message with username and message and sends it over the socket connection
         * listens for a return message and logs the message if any received
         */
        Socket socket;
        String formattedMessage = username + "::" + message;

        try {
            Log.d("MYSOCKET","trying address: "+addr+" and port: "+Integer.toString(port)+" msg: "+formattedMessage);
            socket = new Socket(addr,port);
            Log.d("MYSOCKET","created socket successfully");
            if (socket.isConnected()){
                //send message to server
                Log.d("MYSOCKET","sending message! ");
                OutputStream out = socket.getOutputStream();
                PrintWriter outWriter = new PrintWriter(out);
                outWriter.print(formattedMessage);
                outWriter.flush();

                //listen for status (error/success) message from server
                char[] buffer = new char[2048];
                int charsRead = 0;
                BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(socket.getInputStream()));
                charsRead = bufferedReader.read(buffer);
                if (charsRead > 0) {
                    final String returnMessage = new String(buffer).substring(0, charsRead);
                    Log.d("MYSOCKET",returnMessage);
                    activity.runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            Toast toast = Toast.makeText(activity,returnMessage, Toast.LENGTH_LONG);
                            toast.show();
                        }
                    });
                } else {
                    Log.d("MYSOCKET","no return message");
                }


                outWriter.close();
                out.flush();
                out.close();
                bufferedReader.close();
            }

        } catch (Exception e) {
            e.printStackTrace();
            Log.d("MYSOCKET","failed");
            activity.runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    Toast toast = Toast.makeText(activity, "socket connection failed.", Toast.LENGTH_LONG);
                    toast.show();
                }
            });
        }
        return null;
    }
}
