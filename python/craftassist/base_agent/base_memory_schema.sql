-- Copyright (c) Facebook, Inc. and its affiliates.


PRAGMA foreign_keys = ON;


CREATE TABLE Memories (
    uuid                    NCHAR(36)       PRIMARY KEY,
    tabl                    VARCHAR(255)    NOT NULL,
    create_time             INTEGER         NOT NULL,
    updated_time            INTEGER         NOT NULL,
    attended_time           INTEGER         NOT NULL DEFAULT 0,
    is_snapshot             BOOLEAN         NOT NULL DEFAULT FALSE
);


CREATE TABLE Chats (
    uuid    NCHAR(36)       PRIMARY KEY,
    speaker VARCHAR(255)    NOT NULL,
    chat    TEXT            NOT NULL,
    time    INTEGER         NOT NULL,

    FOREIGN KEY(uuid) REFERENCES Memories(uuid) ON DELETE CASCADE
);
CREATE INDEX ChatsTime ON Chats(time);



CREATE TABLE Locations (
    uuid    NCHAR(36)       PRIMARY KEY,
    x       FLOAT           NOT NULL,
    y       FLOAT           NOT NULL,
    z       FLOAT           NOT NULL,

    FOREIGN KEY(uuid) REFERENCES Memories(uuid) ON DELETE CASCADE
);
CREATE INDEX LocationsXYZ ON Locations(x, y, z);


CREATE TABLE Times (
    uuid    NCHAR(36)       PRIMARY KEY,
    time    INTEGER           NOT NULL,

    FOREIGN KEY(uuid) REFERENCES Memories(uuid) ON DELETE CASCADE
);


CREATE TABLE Tasks (
    uuid        NCHAR(36)       PRIMARY KEY,
    action_name VARCHAR(32)     NOT NULL,
    pickled     BLOB            NOT NULL,
    paused      BOOLEAN         NOT NULL DEFAULT 0,
    created_at  INTEGER         NOT NULL,
    finished_at INTEGER         NOT NULL DEFAULT -1,

    FOREIGN KEY(uuid) REFERENCES Memories(uuid) ON DELETE CASCADE
);
CREATE INDEX TasksFinishedAt ON Tasks(finished_at);


--TODO: rename this OtherAgents
CREATE TABLE Players (
    uuid        NCHAR(36)       PRIMARY KEY,
    eid         INTEGER,
    x           FLOAT,
    y           FLOAT,
    z           FLOAT,
    yaw         FLOAT,
    pitch       FLOAT,
    name        VARCHAR(255),

    UNIQUE(eid),
    FOREIGN KEY(uuid) REFERENCES Memories(uuid) ON DELETE CASCADE
);
CREATE INDEX PlayersName ON Players(name);


CREATE TABLE Dances (
    uuid      NCHAR(36)       PRIMARY KEY
);


CREATE TABLE Triples (
    uuid        NCHAR(36)       PRIMARY KEY,
    subj        NCHAR(36)       NOT NULL,  -- memid of subj, could be BlockObject, Chat, or other
    pred        VARCHAR(255)    NOT NULL,  -- has_tag_, has_name_, etc.
    obj         TEXT            NOT NULL,
    confidence  FLOAT           NOT NULL DEFAULT 1.0,

    UNIQUE(subj, pred, obj) ON CONFLICT REPLACE,
    FOREIGN KEY(subj) REFERENCES Memories(uuid) ON DELETE CASCADE
);
CREATE INDEX TriplesSubjPred ON Triples(subj, pred);
CREATE INDEX TriplesPredObj ON Triples(pred, obj);

CREATE TABLE SetMems(
    uuid    NCHAR(36)       PRIMARY KEY,
    FOREIGN KEY(uuid) REFERENCES Memories(uuid) ON DELETE CASCADE
);
