-- Copyright (c) Facebook, Inc. and its affiliates.


CREATE TABLE BlockTypes (
    uuid            NCHAR(36)   NOT NULL,
    type_name       VARCHAR(32) NOT NULL,
    bid             TINYINT     NOT NULL,
    meta            TINYINT     NOT NULL,

    FOREIGN KEY(uuid) REFERENCES Memories(uuid) ON DELETE CASCADE
);



CREATE TABLE MobTypes (
    uuid            NCHAR(36)   NOT NULL,
    type_name       VARCHAR(32) NOT NULL,
    bid             TINYINT     NOT NULL,
    meta            TINYINT     NOT NULL,

    FOREIGN KEY(uuid) REFERENCES Memories(uuid) ON DELETE CASCADE
);



-- TODO player_placed is string with player id
CREATE TABLE BlockObjects (
    uuid            NCHAR(36)   NOT NULL,
    x               INTEGER     NOT NULL,
    y               INTEGER     NOT NULL,
    z               INTEGER     NOT NULL,
    bid             TINYINT     NOT NULL,
    meta            TINYINT     NOT NULL,
    agent_placed    BOOLEAN     NOT NULL,
    player_placed   BOOLEAN     NOT NULL,
    updated         INTEGER     NOT NULL,

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


-- special table here so we can keep the unique constraint on block objects
-- TODO remove all unique constraints and only have component objects?
-- TODO player_placed is string with player id
CREATE TABLE ArchivedBlockObjects (
    uuid            NCHAR(36)   NOT NULL,
    x               INTEGER     NOT NULL,
    y               INTEGER     NOT NULL,
    z               INTEGER     NOT NULL,
    bid             TINYINT     NOT NULL,
    meta            TINYINT     NOT NULL,
    agent_placed    BOOLEAN     NOT NULL,
    player_placed   BOOLEAN     NOT NULL,
    updated         INTEGER     NOT NULL,

    FOREIGN KEY(uuid) REFERENCES Memories(uuid) ON DELETE CASCADE
);
CREATE INDEX ArchivedBlockObjectsXYZ ON ArchivedBlockObjects(x, y, z);
CREATE TRIGGER ArchivedBlockObjectsDelete AFTER DELETE ON ArchivedBlockObjects
    WHEN ((SELECT COUNT(*) FROM ArchivedBlockObjects WHERE uuid=OLD.uuid LIMIT 1) == 0)
    BEGIN DELETE FROM Memories WHERE uuid=OLD.uuid;
END; -- delete memory when last block is removed
CREATE TRIGGER ArchivedBlockObjectsUpdate AFTER UPDATE ON ArchivedBlockObjects
    WHEN ((SELECT COUNT(*) FROM ArchivedBlockObjects WHERE uuid=OLD.uuid LIMIT 1) == 0)
    BEGIN DELETE FROM Memories WHERE uuid=OLD.uuid;
END; -- delete memory when last block is removed




CREATE TABLE ComponentObjects (
    uuid            NCHAR(36)   NOT NULL,
    x               INTEGER     NOT NULL,
    y               INTEGER     NOT NULL,
    z               INTEGER     NOT NULL,
    bid             TINYINT     NOT NULL,
    meta            TINYINT     NOT NULL,
    agent_placed    BOOLEAN     NOT NULL,
    player_placed   BOOLEAN     NOT NULL,
    updated         INTEGER     NOT NULL,

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


CREATE TABLE InstSeg(
    uuid            NCHAR(36)   NOT NULL,
    x               INTEGER     NOT NULL,
    y               INTEGER     NOT NULL,
    z               INTEGER     NOT NULL,

    FOREIGN KEY(uuid) REFERENCES Memories(uuid) ON DELETE CASCADE
);
CREATE INDEX InstSegXYZ ON InstSeg(x, y, z);


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


CREATE TABLE Mobs (
    uuid          NCHAR(36)       PRIMARY KEY,
    eid           INTEGER         NOT NULL,
    x             FLOAT           NOT NULL,
    y             FLOAT           NOT NULL,
    z             FLOAT           NOT NULL,
    mobtype       VARCHAR(255)    NOT NULL,
    player_placed BOOLEAN         NOT NULL,
    agent_placed  BOOLEAN         NOT NULL,
    spawn         INTEGER         NOT NULL,

--    UNIQUE(eid), not unique because of archives.  make a MobArchive table?
    FOREIGN KEY(uuid) REFERENCES Memories(uuid) ON DELETE CASCADE
);

CREATE TABLE Rewards (
    uuid    NCHAR(36)       PRIMARY KEY,
    value   VARCHAR(32)     NOT NULL, -- {POSITIVE, NEGATIVE}
    time    INTEGER         NOT NULL
);
