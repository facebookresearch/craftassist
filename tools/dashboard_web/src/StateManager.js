import io from "socket.io-client";
import Memory2D from "./components/Memory2D";
import LiveImage from "./components/LiveImage";
import MainPane from "./MainPane";
import Settings from "./components/Settings";
import LiveObjects from "./components/LiveObjects";
import LiveHumans from "./components/LiveHumans";
import QuerySemanticParser from "./components/QuerySemanticParser";

/**
 * The main state manager for the dashboard.
 * It connects to the backend via socket.io.
 * It drives all frontend components with new data from the backend.
 * It drives all backend commands from the frontend.
 * It persistently reconnects to the backend upon disconnection.
 *
 * The interface to the frontend is mainly via two methods:
 * 1. connect(c): a UI component can connect to the stateManager
 *                and the stateManager will subsequently handle state
 *                updates to this UI component
 * 2. restart(url): the stateManager can flush out and restart itself to
 *                  a different URL than default
 *
 * The interface to the backend is via socket.io events.
 * The main event of interests are:
 * "sensor_payload": this is a message that is received from the backend
 *                   with the latest sensor metadata. The stateManager
 *                   subsequently updates the frontend with this metadata.
 * "command": this is a raw command message that is sent to the backend
 *            (currently in Navigator.js, but should be
 *             refactored to be in this class).
 *
 * This class can be seen as a poor-man's Redux.js library, as Redux
 * would be overkill for this small project.
 */
class StateManager {
  refs = [];
  socket = null;
  default_url = "http://localhost:8000";
  connected = false;
  memory = {
    objects: new Map(),
    humans: new Map(),
  };
  constructor() {
    this.updateMemory = this.updateMemory.bind(this);
    this.processSensorPayload = this.processSensorPayload.bind(this);
    this.querySemanticParser = this.querySemanticParser.bind(this);
    this.setConnected = this.setConnected.bind(this);
    this.setImageSettings = this.setImageSettings.bind(this);
    this.keyHandler = this.keyHandler.bind(this);

    let url = localStorage.getItem("server_url");
    if (url === "undefined" || url === undefined || url === null) {
      url = this.default_url;
    }
    this.setUrl(url);

    this.fps_time = performance.now();
  }

  setDefaultUrl() {
    localStorage.clear();
    this.setUrl(this.default_url);
  }

  setUrl(url) {
    this.url = url;
    localStorage.setItem("server_url", url);
    this.restart(this.url);
  }

  restart(url) {
    this.socket = io.connect(url, {
      transports: ["polling", "websocket"],
    });
    const socket = this.socket;
    // on reconnection, reset the transports option, as the Websocket
    // connection may have failed (caused by proxy, firewall, browser, ...)
    socket.on("reconnect_attempt", () => {
      socket.io.opts.transports = ["polling", "websocket"];
    });

    socket.on("after connect", (msg) => {
      this.setConnected(true);
    });

    socket.on("disconnect", (msg) => {
      this.setConnected(false);
      console.log("disconnected");
    });

    socket.on("render_parser_output", this.querySemanticParser);
    socket.on("sensor_payload", this.processSensorPayload);
    socket.on("image_settings", this.setImageSettings);
  }

  setImageSettings(newSettings) {
    this.refs.forEach((ref) => {
      if (ref instanceof Settings) {
        ref.setState(newSettings);
      }
    });
  }

  setConnected(status) {
    this.connected = status;
    this.refs.forEach((ref) => {
      if (ref instanceof Settings) {
        ref.setState({ connected: status });
      }
    });
  }

  updateMemory(res) {
    res.objects.forEach((obj) => {
      let key = JSON.stringify(obj); // I'm horrible person for doing this!!!
      obj.xyz = [obj.xyz[0] + res.x, obj.xyz[1] + res.y, obj.xyz[2]];
      this.memory.objects.set(key, obj);
    });
  }

  querySemanticParser(res) {
    this.refs.forEach((ref) => {
      if (ref instanceof QuerySemanticParser) {
        console.log("now setting state with " + res.action_dict);
        ref.setState({
          action_dict: res.action_dict,
        });
      }
    });
  }

  keyHandler(key_codes) {
    let commands = [];
    for (var k in key_codes) {
      let val = key_codes[k];
      k = parseInt(k);
      if (val === true) {
        if (k === 38) {
          // Up
          commands.push("MOVE_FORWARD");
        }
        if (k === 40) {
          // Down
          commands.push("MOVE_BACKWARD");
        }
        if (k === 37) {
          // Left
          commands.push("MOVE_LEFT");
        }
        if (k === 39) {
          // Right
          commands.push("MOVE_RIGHT");
        }
      }
    }
    if (commands.length > 0) {
      this.socket.emit("command", commands);
    }
  }

  processSensorPayload(res) {
    let fps_time = performance.now();
    let fps = 1000 / (fps_time - this.fps_time);
    this.fps_time = fps_time;
    let rgb = new Image();
    rgb.src = "data:image/webp;base64," + res.image.rgb;
    let depth = new Image();
    depth.src = "data:image/webp;base64," + res.image.depth;
    let object_rgb = new Image();
    if (res.object_image != -1) {
      object_rgb.src = "data:image/webp;base64," + res.object_image.rgb;
    }

    this.updateMemory(res);

    this.refs.forEach((ref) => {
      if (ref instanceof Memory2D) {
        ref.setState({
          isLoaded: true,
          memory: this.memory,
          bot_xyz: [res.x, res.y, res.yaw],
        });
      } else if (ref instanceof Settings) {
        ref.setState({ fps: fps });
      } else if (ref instanceof LiveImage) {
        ref.setState({
          isLoaded: true,
          rgb: rgb,
          depth: depth,
        });
      } else if (ref instanceof LiveObjects || ref instanceof LiveHumans) {
        if (res.object_image != -1) {
          ref.setState({
            isLoaded: true,
            rgb: object_rgb,
            objects: res.objects,
            humans: res.humans,
          });
        }
      }
    });
    return "OK";
  }

  connect(o) {
    this.refs.push(o);
  }
}
var stateManager = new StateManager();

// export a single reused stateManager object,
// rather than the class, so that it is reused across tests in the same lifetime
export default stateManager;
