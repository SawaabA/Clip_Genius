-- Create Sports Table
CREATE TABLE
    Sports (
        SportID INT AUTO_INCREMENT PRIMARY KEY,
        Name VARCHAR(100) NOT NULL,
        CreatedDate DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        INDEX (Name)
    );

-- Create Users Table
CREATE TABLE
    Users (
        UserID INT AUTO_INCREMENT PRIMARY KEY,
        Username VARCHAR(100) NOT NULL UNIQUE,
        Email VARCHAR(255) NOT NULL UNIQUE,
        PasswordHash VARCHAR(255) NOT NULL,
        RegistrationDate DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        LastLoginDate DATETIME,
        Role ENUM ('admin', 'viewer') NOT NULL,
        INDEX (Username),
        INDEX (Email)
    );

-- Create Matches Table
CREATE TABLE
    Matches (
        MatchID INT AUTO_INCREMENT PRIMARY KEY,
        Date DATE NOT NULL,
        SportID INT NOT NULL,
        Team1Name VARCHAR(255) NOT NULL,
        Team2Name VARCHAR(255) NOT NULL,
        FOREIGN KEY (SportID) REFERENCES Sports (SportID),
        INDEX (Date),
    );

-- Create Videos Table
CREATE TABLE
    Videos (
        VideoID INT AUTO_INCREMENT PRIMARY KEY,
        MatchID INT NOT NULL,
        VideoFilePath VARCHAR(1024) NOT NULL,
        ThumbnailPath VARCHAR(1024),
        Duration INT,
        FileSize BIGINT, -- Size in bytes
        CreatedDate DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        Views BIGINT DEFAULT 0,
        FOREIGN KEY (MatchID) REFERENCES Matches (MatchID),
        INDEX (CreatedDate)
    );

-- Create Tags Table
CREATE TABLE
    Tags (
        TagID INT AUTO_INCREMENT PRIMARY KEY,
        TagName VARCHAR(100) NOT NULL UNIQUE
    );

-- Create VideoTags Table
CREATE TABLE
    VideoTags (
        VideoTagID INT AUTO_INCREMENT PRIMARY KEY,
        VideoID INT NOT NULL,
        TagID INT NOT NULL,
        FOREIGN KEY (VideoID) REFERENCES Videos (VideoID),
        FOREIGN KEY (TagID) REFERENCES Tags (TagID),
        INDEX (TagID)
    );

-- Create Comments Table
CREATE TABLE
    Comments (
        CommentID INT AUTO_INCREMENT PRIMARY KEY,
        VideoID INT NOT NULL,
        UserID INT NOT NULL,
        Comment TEXT NOT NULL,
        CommentDate DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        ModifiedDate DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        FOREIGN KEY (VideoID) REFERENCES Videos (VideoID) ON DELETE CASCADE,
        FOREIGN KEY (UserID) REFERENCES Users (UserID) ON DELETE CASCADE,
        FULLTEXT (Comment),
        INDEX (CommentDate)
    );

-- Create Ratings Table
CREATE TABLE
    Ratings (
        RatingID INT AUTO_INCREMENT PRIMARY KEY,
        VideoID INT NOT NULL,
        UserID INT NOT NULL,
        Rating TINYINT NOT NULL CHECK (Rating BETWEEN 1 AND 5),
        RatingDate DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (VideoID) REFERENCES Videos (VideoID),
        FOREIGN KEY (UserID) REFERENCES Users (UserID),
        INDEX (RatingDate)
    );