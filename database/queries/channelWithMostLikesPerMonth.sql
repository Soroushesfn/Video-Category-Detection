WITH
-- 1) parse the “Weekday, D Month YYYY” into an ISO date
parsed AS (
  SELECT
    channel_title,
    likes,
    trim(substr(publish_date, instr(publish_date, ',')+1)) AS pd
  FROM US_Trending_Videos
),
split AS (
  SELECT
    channel_title,
    likes,
    cast(substr(pd, 1, instr(pd,' ')-1) AS integer)       AS day,
    substr(
      pd,
      instr(pd,' ')+1,
      instr(substr(pd, instr(pd,' ')+1), ' ') - 1
    )                                                     AS month_name,
    cast(substr(pd, -4) AS integer)                       AS year
  FROM parsed
),
iso AS (
  SELECT
    channel_title,
    likes,
    printf('%04d-%02d-%02d',
      year,
      CASE month_name
        WHEN 'January'   THEN 1 WHEN 'February' THEN 2
        WHEN 'March'     THEN 3 WHEN 'April'    THEN 4
        WHEN 'May'       THEN 5 WHEN 'June'     THEN 6
        WHEN 'July'      THEN 7 WHEN 'August'   THEN 8
        WHEN 'September' THEN 9 WHEN 'October'  THEN 10
        WHEN 'November'  THEN 11 WHEN 'December' THEN 12
      END,
      day
    ) AS iso_date
  FROM split
),

-- 2) sum likes by channel & month
monthly_likes AS (
  SELECT
    strftime('%Y-%m', iso_date)     AS month,
    channel_title,
    SUM(likes)                      AS total_likes
  FROM iso
  GROUP BY month, channel_title
),

-- 3) rank channels within each month by total_likes
ranked AS (
  SELECT
    month,
    channel_title,
    total_likes,
    ROW_NUMBER() OVER (
      PARTITION BY month
      ORDER BY total_likes DESC
    ) AS rn
  FROM monthly_likes
)

-- 4) pick the top-ranked (rn=1) per month
SELECT
  month,
  channel_title    AS top_channel,
  total_likes
FROM ranked
WHERE rn = 1
ORDER BY month;
