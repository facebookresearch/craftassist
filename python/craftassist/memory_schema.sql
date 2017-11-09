-- Copyright (c) Facebook, Inc. and its affiliates.


PRAGMA foreign_keys = ON;


CREATE TABLE Memories (
    uuid                    NCHAR(36)       PRIMARY KEY,
    tabl                    VARCHAR(255)    NOT NULL,
    workspace_updated_time  FLOAT           NOT NULL DEFAULT 0
);



CREATE TABLE BlockObjects (
    uuid            NCHAR(36)   NOT NULL,
    x               INTEGER     NOT NULL,
    y               INTEGER     NOT NULL,
    z               INTEGER     NOT NULL,
    bid             TINYINT     NOT NULL,
    meta            TINYINT     NOT NULL,
    player_placed   BOOLEAN     NOT NULL,
    updated         FLOAT       NOT NULL,

    UNIQUE(x, y, z),  -- remove for components
    FOREIGN KEY(uuid) REFERENCES Memories(uuid) ON DELETE CASCADE
);
CREATE INDEX BlockObjectsXYZ ON BlockObjects(x, y, z);
CREATE TRIGGER BlockObjectsDelete AFTER DELETE ON BlockObjects
    WHEN ((SELECT COUNT(*) FROM BlockObjects WHERE uuid=OLD.uuid LIMIT 1) == 0)
    BEGIN DELETE FROM Memories WHERE uuid=OLD.uuid;
END; -- delete memory when last block is removed
CREATE TRIGGER BlockObjectsUpdate AFTER UPDATE ON BlockObjects
    WHEN ((SELECT COUNT(*) FROM BlockObjects WHERE uuid=OLD.uuid LIMIT 1) == 0)
    BEGIN DELETE FROM Memories WHERE uuid=OLD.uuid;
END; -- delete memory when last block is removed
CREATE TRIGGER BlockObjectsWorkspaceUpd AFTER UPDATE ON BlockObjects
    WHEN NEW.player_placed == 1
    BEGIN UPDATE Memories SET workspace_updated_time=strftime('%s','now') WHERE uuid=NEW.uuid;
END;  -- update workspace memory
CREATE TRIGGER BlockObjectsWorkspaceIns AFTER INSERT ON BlockObjects
    WHEN NEW.player_placed == 1
    BEGIN UPDATE Memories SET workspace_updated_time=strftime('%s','now') WHERE uuid=NEW.uuid;
END;  -- update workspace memory



CREATE TABLE ComponentObjects (
    uuid            NCHAR(36)   NOT NULL,
    x               INTEGER     NOT NULL,
    y               INTEGER     NOT NULL,
    z               INTEGER     NOT NULL,
    bid             TINYINT     NOT NULL,
    meta            TINYINT     NOT NULL,
    player_placed   BOOLEAN     NOT NULL,
    updated         FLOAT       NOT NULL,

    -- UNIQUE(x, y, z),  -- remove for components
    FOREIGN KEY(uuid) REFERENCES Memories(uuid) ON DELETE CASCADE
);
CREATE INDEX ComponentObjectsXYZ ON ComponentObjects(x, y, z);
CREATE TRIGGER ComponentObjectsDelete AFTER DELETE ON ComponentObjects
    WHEN ((SELECT COUNT(*) FROM ComponentObjects WHERE uuid=OLD.uuid LIMIT 1) == 0)
    BEGIN DELETE FROM Memories WHERE uuid=OLD.uuid;
END; -- delete memory when last block is removed
CREATE TRIGGER ComponentObjectsUpdate AFTER UPDATE ON ComponentObjects
    WHEN ((SELECT COUNT(*) FROM ComponentObjects WHERE uuid=OLD.uuid LIMIT 1) == 0)
    BEGIN DELETE FROM Memories WHERE uuid=OLD.uuid;
END; -- delete memory when last block is removed
CREATE TRIGGER ComponentObjectsWorkspaceUpd AFTER UPDATE ON ComponentObjects
    WHEN NEW.player_placed == 1
    BEGIN UPDATE Memories SET workspace_updated_time=strftime('%s','now') WHERE uuid=NEW.uuid;
END;  -- update workspace memory
CREATE TRIGGER ComponentObjectsWorkspaceIns AFTER INSERT ON ComponentObjects
    WHEN NEW.player_placed == 1
    BEGIN UPDATE Memories SET workspace_updated_time=strftime('%s','now') WHERE uuid=NEW.uuid;
END;  -- update workspace memory


CREATE TABLE InstSeg(
    uuid            NCHAR(36)   NOT NULL,
    x               INTEGER     NOT NULL,
    y               INTEGER     NOT NULL,
    z               INTEGER     NOT NULL,

    FOREIGN KEY(uuid) REFERENCES Memories(uuid) ON DELETE CASCADE
);
CREATE INDEX InstSegXYZ ON InstSeg(x, y, z);


CREATE TABLE ArchivedInstSeg (
    uuid            NCHAR(36)   NOT NULL,
    x               INTEGER     NOT NULL,
    y               INTEGER     NOT NULL,
    z               INTEGER     NOT NULL,

    FOREIGN KEY(uuid) REFERENCES Memories(uuid) ON DELETE CASCADE
);
CREATE INDEX ArchivedInstSegXYZ ON ArchivedInstSeg(x, y, z);


CREATE TABLE Schematics (
    uuid    NCHAR(36)   NOT NULL,
    x       INTEGER     NOT NULL,
    y       INTEGER     NOT NULL,
    z       INTEGER     NOT NULL,
    bid     TINYINT     NOT NULL,
    meta    TINYINT     NOT NULL,

    UNIQUE(uuid, x, y, z),
    FOREIGN KEY(uuid) REFERENCES Memories(uuid) ON DELETE CASCADE
);
CREATE INDEX SchematicsXYZ ON Schematics(x, y, z);


CREATE TABLE Chats (
    uuid    NCHAR(36)       PRIMARY KEY,
    speaker VARCHAR(255)    NOT NULL,
    chat    TEXT            NOT NULL,
    time    FLOAT           NOT NULL,

    FOREIGN KEY(uuid) REFERENCES Memories(uuid) ON DELETE CASCADE
);
CREATE INDEX ChatsTime ON Chats(time);


CREATE TABLE Mobs (
    uuid    NCHAR(36)       PRIMARY KEY,
    eid     INTEGER         NOT NULL,
    x       FLOAT           NOT NULL,
    y       FLOAT           NOT NULL,
    z       FLOAT           NOT NULL,
    mobtype VARCHAR(255)    NOT NULL,
    spawn   FLOAT           NOT NULL,

    UNIQUE(eid),
    FOREIGN KEY(uuid) REFERENCES Memories(uuid) ON DELETE CASCADE
);


CREATE TABLE Locations (
    uuid    NCHAR(36)       PRIMARY KEY,
    x       FLOAT           NOT NULL,
    y       FLOAT           NOT NULL,
    z       FLOAT           NOT NULL,

    FOREIGN KEY(uuid) REFERENCES Memories(uuid) ON DELETE CASCADE
);
CREATE INDEX LocationsXYZ ON Locations(x, y, z);


CREATE TABLE Tasks (
    uuid        NCHAR(36)       PRIMARY KEY,
    action_name VARCHAR(32)     NOT NULL,
    pickled     BLOB            NOT NULL,
    paused      BOOLEAN         NOT NULL DEFAULT 0,
    created_at  FLOAT           NOT NULL,
    finished_at FLOAT           NOT NULL DEFAULT -1,

    FOREIGN KEY(uuid) REFERENCES Memories(uuid) ON DELETE CASCADE
);
CREATE INDEX TasksFinishedAt ON Tasks(finished_at);


CREATE TABLE Players (
    uuid      NCHAR(36)       PRIMARY KEY,
    name      VARCHAR(255)    NOT NULL,

    UNIQUE(name),
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


CREATE TABLE Rewards (
    uuid    NCHAR(36)       PRIMARY KEY,
    value   VARCHAR(32)     NOT NULL, -- {POSITIVE, NEGATIVE}
    time    FLOAT           NOT NULL
);


CREATE TABLE SetMems(
    uuid    NCHAR(36)       PRIMARY KEY,
    time    FLOAT           NOT NULL,

    FOREIGN KEY(uuid) REFERENCES Memories(uuid) ON DELETE CASCADE
);
