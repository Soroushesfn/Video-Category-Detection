
SELECT
  channel_title,
  COUNT(
    
      (substr(trending_date,7,4) || '-' ||
       substr(trending_date,4,2) || '-' ||
       substr(trending_date,1,2))
  ) AS trending_days
FROM US_Trending_Videos
GROUP BY channel_title
ORDER BY trending_days DESC;