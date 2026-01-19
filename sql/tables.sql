CREATE TABLE IF NOT EXISTS youtube_videos (
    video_id VARCHAR(50) PRIMARY KEY,
    title TEXT,
    published_at TIMESTAMP,
    channel_id VARCHAR(50),
    channel_title TEXT,
    search_query TEXT,
    processed_date DATE,
    channel_country VARCHAR(5)
);


CREATE TABLE IF NOT EXISTS youtube_comments (
    comment_id VARCHAR(50) PRIMARY KEY,
    video_id VARCHAR(50) REFERENCES youtube_videos(video_id),
    text_display TEXT,
    author_name TEXT,
    published_at TIMESTAMP,
    like_count INT,
    processed_date DATE
);


CREATE TABLE IF NOT EXISTS comment_sentiment (
    comment_id VARCHAR(50) PRIMARY KEY REFERENCES youtube_comments(comment_id),
    label VARCHAR(20),
    score FLOAT,
    model_name VARCHAR(100),
    processed_date DATE,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);