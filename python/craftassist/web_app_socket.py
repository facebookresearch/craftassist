import logging
import socket
import json
import os

HOST = "127.0.0.1"  # The server's hostname or IP address
PORT = 2557  # The port used by the server

dirname = os.path.dirname(__file__)
web_app_filename = os.path.join(dirname, "webapp_data.json")

# create a socket
s = socket.socket()
print("web app socket: created")

# s.bind(('', PORT)) specifies that the socket is reachable by any
# address the machine happens to have.
s.bind(("", PORT))

# become a server socket and queue as many as 5 requests before
# refusing connections
s.listen(5)


def convert(dictionary):
    """Recursively converts dictionary keys to strings."""
    if not isinstance(dictionary, dict):
        return dictionary
    return dict((str(k), convert(v)) for k, v in dictionary.items())


while True:
    conn, addr = s.accept()
    print("web app socket: got connection from ", addr)

    # receive bytes and decode to string
    msgrec = conn.recv(10000).decode()
    print("web app socket: received ", msgrec)

    # here add hooks to decide what to do with the data / message
    # 1. query dialogue_manager /
    # read the file to return action dict
    try:
        with open(web_app_filename) as f:
            data = json.load(f)
            print(data)
            try:
                data = convert(data)
            except Exception as e:
                print(e)

            print("sending")
            print(data[msgrec])
            if msgrec in data:
                conn.send((json.dumps(data[msgrec])).encode())
            else:
                # send back empty data if not message not found
                conn.send((str({})).encode())
    except:
        conn.send((str({})).encode())

    logging.info("web app socket: closing")
    conn.close()
