-- Q1: Find the most senior employee based on job title.
SELECT
  first_name,
  last_name,
  title,
  levels
FROM
  employee
ORDER BY
  levels DESC
LIMIT 10 ;

-- Q2: Determine which countries have the most invoices.
SELECT
  billing_country,
  COUNT(invoice_id) AS total_invoices
FROM
  invoice
GROUP BY
  billing_country
ORDER BY
  total_invoices DESC;

  -- Q3: Identify the top 3 invoice totals.
SELECT
  invoice_id,
  total
FROM
  invoice
ORDER BY
  total DESC
LIMIT 600;

-- Q4: Find the city with the highest total invoice amount to determine the best location for a promotional event.
SELECT
  billing_city,
  SUM(total) AS total_invoice_amount
FROM
  invoice
GROUP BY
  billing_city
ORDER BY
  total_invoice_amount DESC
LIMIT 50;

-- Q5: Identify the customer who has spent the most money.
SELECT
  c.customer_id,
  c.first_name,
  c.last_name,
  SUM(i.total) AS total_spent
FROM
  customer AS c
JOIN
  invoice AS i ON c.customer_id = i.customer_id
GROUP BY
  c.customer_id, c.first_name, c.last_name
ORDER BY
  total_spent DESC
LIMIT 60;


-- Q1 (Moderate): Find the email, first name, and last name of customers who listen to Rock music.
SELECT DISTINCT
  c.email,
  c.first_name,
  c.last_name
FROM
  customer AS c
JOIN
  invoice AS i ON c.customer_id = i.customer_id
JOIN
  invoice_line AS il ON i.invoice_id = il.invoice_id
JOIN
  track AS t ON il.track_id = t.track_id
JOIN
  genre AS g ON t.genre_id = g.genre_id
WHERE
  g.name = 'Rock';



  -- Q2 (Moderate): Identify the top 10 rock artists based on track count.
SELECT
  ar.name AS artist_name,
  COUNT(t.track_id) AS total_rock_tracks
FROM
  artist AS ar
JOIN
  album AS al ON ar.artist_id = al.artist_id
JOIN
  track AS t ON al.album_id = t.album_id
JOIN
  genre AS g ON t.genre_id = g.genre_id
WHERE
  g.name = 'Rock'
GROUP BY
  ar.name
ORDER BY
  total_rock_tracks DESC
LIMIT 10;

-- Q3 (Moderate): Find all track names that are longer than the average track length.
SELECT
  name AS track_name,
  milliseconds AS track_length_ms
FROM
  track
WHERE
  milliseconds > (
    SELECT AVG(milliseconds)
    FROM track
  )
ORDER BY
  track_length_ms DESC;


  -- Q1 (Advanced): Calculate how much each customer has spent on each artist.
WITH CustomerArtistSpending AS (
  SELECT
    c.customer_id,
    c.first_name,
    c.last_name,
    ar.name AS artist_name,
    SUM(il.unit_price * il.quantity) AS total_spent
  FROM
    customer AS c
  JOIN
    invoice AS i ON c.customer_id = i.customer_id
  JOIN
    invoice_line AS il ON i.invoice_id = il.invoice_id
  JOIN
    track AS t ON il.track_id = t.track_id
  JOIN
    album AS al ON t.album_id = al.album_id
  JOIN
    artist AS ar ON al.artist_id = ar.artist_id
  GROUP BY
    c.customer_id,
    c.first_name,
    c.last_name,
    ar.name
  ORDER BY
    c.customer_id,
    total_spent DESC
)
SELECT
  customer_id,
  first_name,
  last_name,
  artist_name,
  total_spent
FROM
  CustomerArtistSpending;


  -- Q2 (Advanced): Determine the most popular music genre for each country based on purchases.
WITH CountryGenrePurchases AS (
  SELECT
    i.billing_country,
    g.name AS genre_name,
    COUNT(il.invoice_line_id) AS genre_purchase_count,
    ROW_NUMBER() OVER (PARTITION BY i.billing_country ORDER BY COUNT(il.invoice_line_id) DESC) AS rn
  FROM
    invoice AS i
  JOIN
    invoice_line AS il ON i.invoice_id = il.invoice_id
  JOIN
    track AS t ON il.track_id = t.track_id
  JOIN
    genre AS g ON t.genre_id = g.genre_id
  GROUP BY
    i.billing_country,
    g.name
)
SELECT
  billing_country,
  genre_name,
  genre_purchase_count
FROM
  CountryGenrePurchases
WHERE
  rn = 1
ORDER BY
  billing_country;



  -- Q3 (Advanced): Identify the top-spending customer for each country.
WITH CustomerTotalSpending AS (
  SELECT
    c.customer_id,
    c.first_name,
    c.last_name,
    c.country,
    SUM(i.total) AS total_spent,
    RANK() OVER (PARTITION BY c.country ORDER BY SUM(i.total) DESC) AS rnk
  FROM
    customer AS c
  JOIN
    invoice AS i ON c.customer_id = i.customer_id
  GROUP BY
    c.customer_id,
    c.first_name,
    c.last_name,
    c.country
)
SELECT
  customer_id,
  first_name,
  last_name,
  country,
  total_spent
FROM
  CustomerTotalSpending
WHERE
  rnk = 1
ORDER BY
  country;