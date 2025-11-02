WITH RECURSIVE
-- 1) split the pipe-delimited tags into one row per tag
split(channel_title, tag, rest) AS (
  SELECT
    channel_title,
    CASE 
      WHEN instr(tags, '|') > 0 
        THEN substr(tags, 1, instr(tags, '|')-1) 
      ELSE tags 
    END,
    CASE
      WHEN instr(tags, '|') > 0
        THEN substr(tags, instr(tags, '|')+1)
      ELSE ''
    END
  FROM US_Trending_Videos

  UNION ALL

  SELECT
    channel_title,
    CASE
      WHEN instr(rest, '|') > 0 
        THEN substr(rest, 1, instr(rest, '|')-1)
      ELSE rest
    END,
    CASE
      WHEN instr(rest, '|') > 0
        THEN substr(rest, instr(rest, '|')+1)
      ELSE ''
    END
  FROM split
  WHERE rest <> ''
),

-- 2) count each tag per channel
counts AS (
  SELECT
    channel_title,
    tag,
    COUNT(*) AS cnt
  FROM split
  WHERE tag <> ''
  GROUP BY channel_title, tag
),

-- 3) find the top count per channel
max_counts AS (
  SELECT
    channel_title,
    MAX(cnt) AS max_cnt
  FROM counts
  GROUP BY channel_title
)

-- 4) select the tag(s) matching that top count, sorted by count desc
SELECT
  c.channel_title,
  c.tag           AS top_tag,
  c.cnt           AS usage_count
FROM counts c
JOIN max_counts m
  ON c.channel_title = m.channel_title
 AND c.cnt           = m.max_cnt
ORDER BY usage_count DESC;