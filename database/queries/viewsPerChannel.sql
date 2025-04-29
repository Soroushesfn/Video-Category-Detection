SELECT 
  channel_title,
  SUM(views) AS total_views
FROM US_Trending_Videos
GROUP BY channel_title
ORDER BY total_views DESC;
