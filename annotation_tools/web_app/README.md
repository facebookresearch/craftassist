# Web App Error Annotation Tool for use with the [Minecraft Project](https://github.com/fairinternal/minecraft)

## Dependencies
You must install NodeJS version 10 or higher and NPM (most installations include this by default) to run this application.

```
# on MacOS
brew install node

# on Ubuntu
# replace setup_14.x with setup_12.x, setup_10.x for Node v12/10
curl -sL https://deb.nodesource.com/setup_14.x | sudo -E bash -
sudo apt-get install -y nodejs
```

After you install, use `node -v` and `npm -v` to check that the installation completed successfully.

## Set Up the Web App

From both `minecraft/annotation_tools/web_app/frontend` and `minecraft/annotation_tools/web_app/backend` run:
```
npm install
```
to install all the dependencies and packages needed by the Web App.

From both `minecraft/annotation_tools/web_app/frontend` and `minecraft/annotation_tools/web_app/backend` again, run:
```
npm start
```
to start up the backend and frontend. The backend is listening on port 9000 and the frontend should be serving the app on port 3000. Open [http://localhost:3000](http://localhost:3000) to view it.

If the above fails try:
```
npm rebuild
```

## Set Up the Project
Do this *before* setting up the Web App.
Go through the project's [installation](https://github.com/fairinternal/minecraft#installation--getting-started) with a few modifications specified below.

**When going through the project setup, please do not use ports 2556 or 2557 as these will be used by the web app** (You can use the default ports in the setup process).

At [Run the Cuberite instance](https://github.com/fairinternal/minecraft#installation--getting-started) step of the project's installation, run the following command instead:
```
python ./python/cuberite_process.py --add-plugin minecraft_asr_app --config flat_world
```
which starts an instance of cuberite listening on `localhost:25565` with the plugin `minecraft_asr_app` enabled.

Continue to follow the rest of the setup instructions from [Connecting your Minecraft game client](https://github.com/fairinternal/minecraft#connecting-your-minecraft-game-client-so-you-can-see-whats-happening).

At [Running the interactive V0 agent](https://github.com/fairinternal/minecraft#running-the-interactive-v0-agent) step, run the command in a separate tab but with a `--web_app` flag just like below.
```
python ./python/craftassist/craftassist_agent.py --web_app
```

## Set up the socket connection
Now in a separate tab, run:
```
cd python/craftassist/
python web_app_socket.py
```
to set up the socket connection between the web application and cuberite.


## About the Web App
With this web app, Minecraft players can interact and communicate with the bot using voice as input. The app uses the [Web Speech API](https://w3c.github.io/speech-api/) to parse voice to text which is then sent as a chat message from the player in the game through the specified IP address and port from the first page of the web app.

Functionality:
- Speech recognition
- Chat history
- Username specified must be a player within the game for the chat message to be sent
- Error annotation tool
- Saves user input (from error annotation) to a local database

## Usage on FAIR cluster
### Connecting the Web App to the `minecraft_asr_app` plugin
If Cuberite is running remotely on a FAIR cluster, create an ssh tunnel for cuberite:
```
ssh -fN -L 25565:100.97.69.35:25565 -J snc-fairjmp101 kavyasrinet@100.97.69.35
```
Now create a tunnel from the port that the minecraft_asr_app plugin of cuberite is listening on :
```
ssh -fN -L 2556:100.97.69.35:2556 -J snc-fairjmp101 kavyasrinet@100.97.69.35
```
This allows the Web App to send chat messages to the bot.

Do the same thing for the port (2557) that the craftassist_agent (with the `--web_app` flag) is using to listen for action dictionary and adtt output requests by running the command below:
```
ssh -fN -L 2557:100.97.69.35:2557 -J snc-fairjmp101 kavyasrinet@100.97.69.35
```
(Replace `100.97.69.35` with your devserver's IP and `kavyasrinet` with your username.)

Running `craftassist_agent.py` with the `--web_app` flag executes a python script in the background that listens on port 2557 to return the action dictionary and adtt model output. After `craftassist_agent.py` finishes running, remember to kill the process that is still running (listening on port 2557) as shown below:


### Using the App
The `minecraft_asr_app` cuberite plugin starts a server which listens on a specified port (default 2556). For the app to send chat messages, it must communicate with the Cuberite plugin.

Connecting: When the Web App loads, the first screen should be the settings page. Input the IP address (`127.0.0.1`), port (`2556`), and your minecraft username that is being used in game. Once you are done click the `submit` button to start communicating with the bot.

Sending a chat in-game: Press the space bar to start recording and then speak the message you would like to send to the bot. You can edit the text or manually type in the message in the textbox. Click `submit` to send the chat (to the `minecraft_asr_app` cuberite plugin which sends it to the Minecraft client). The chat should appear in-game as well as above the textbox. As you send more messages, a small chat history will appear.

Reporting Errors: If the bot responded in any way that you did not expect, press the x button next to the message that resulted in an error. Follow the question flow and answer the questions prompted.
- Correcting the action dictionary table: if the labels and values are correct then hit done. Else correct them by either selecting the right option from the dropdown or editing the text field.

## Developer Notes
The Web App saves the data into the `minecraft/web_app/backend/web_app_data.db` in the `data` table.

### Build for production
The Web App is still in developer mode. To build the app for production, run

`npm run build`

which builds the app for production to the `build` folder.<br>
It correctly bundles React in production mode and optimizes the build for the best performance.

The build is compressed and the filenames include the hashes.<br>
Your app is ready to be deployed!

See the section about [deployment](https://facebook.github.io/create-react-app/docs/deployment) for more information.


## Built with
- The frontend was bootstrapped with [Create React App](https://github.com/facebook/create-react-app).
- The backend uses [sqlite](https://www.sqlite.org/index.html) to save the data to a local database.
