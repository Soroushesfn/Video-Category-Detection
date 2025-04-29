WITH parsed AS (
  SELECT
    views,
    trim(substr(publish_date, instr(publish_date, ',')+1)) AS pd
  FROM US_Trending_Videos
),
split AS (
  SELECT
    views,
    cast(substr(pd, 1, instr(pd,' ')-1) AS integer)            AS day,
    substr(
      pd,
      instr(pd,' ')+1,
      instr(substr(pd, instr(pd,' ')+1), ' ') - 1
    )                                                          AS month_name,
    cast(substr(pd, -4) AS integer)                            AS year
  FROM parsed
),
iso AS (
  SELECT
    views,
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
)
SELECT
  substr(iso_date,6,2)                             AS publish_month,
  CASE substr(iso_date,6,2)
    WHEN '01' THEN 'January'   WHEN '02' THEN 'February'
    WHEN '03' THEN 'March'     WHEN '04' THEN 'April'
    WHEN '05' THEN 'May'       WHEN '06' THEN 'June'
    WHEN '07' THEN 'July'      WHEN '08' THEN 'August'
    WHEN '09' THEN 'September' WHEN '10' THEN 'October'
    WHEN '11' THEN 'November'  WHEN '12' THEN 'December'
  END                                              AS month_name,
  SUM(views)                                       AS total_views
FROM iso
GROUP BY publish_month, month_name
ORDER BY publish_month;
