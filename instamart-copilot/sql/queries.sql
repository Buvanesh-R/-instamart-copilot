-- Part A, task 6: seven analytical queries on data/instamart.db.
-- Unless stated otherwise only completed orders count as revenue or spend.
-- Each query starts with a "-- Qn" marker line; src/a3_sql_queries.py runs them one by one.

-- Q1: WHERE then HAVING. Completed orders above INR 500 (WHERE), grouped by customer,
--     keeping customers whose total of those orders exceeds INR 8000 (HAVING).
SELECT customer_id,
       COUNT(*)                     AS n_orders,
       ROUND(SUM(order_amount), 2)  AS total_spend
FROM orders
WHERE status = 'completed'
  AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY total_spend DESC;

-- Q2: three-table join. Revenue by city and product category.
SELECT c.city,
       p.category,
       ROUND(SUM(o.order_amount), 2) AS revenue
FROM orders    AS o
JOIN customers AS c ON c.customer_id = o.customer_id
JOIN products  AS p ON p.product_id  = o.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY revenue DESC;

-- Q3: scalar subquery. Products priced above the average catalogue price.
SELECT product_id, product_name, category, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC, product_id;

-- Q4: LEFT JOIN. Customers who have no completed order.
SELECT c.customer_id, c.city
FROM customers AS c
LEFT JOIN orders AS o
       ON o.customer_id = c.customer_id
      AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;

-- Q5: common table expression. One CTE holds each customer's total completed spend,
--     a second holds the average of those totals; the main query keeps customers above the average.
WITH customer_spend AS (
    SELECT customer_id, SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
avg_spend AS (
    SELECT AVG(total_spend) AS avg_total_spend
    FROM customer_spend
)
SELECT cs.customer_id,
       ROUND(cs.total_spend, 2)      AS total_spend,
       ROUND(a.avg_total_spend, 2)   AS avg_total_spend
FROM customer_spend AS cs
CROSS JOIN avg_spend AS a
WHERE cs.total_spend > a.avg_total_spend
ORDER BY cs.total_spend DESC;

-- Q6: ranking window functions. Within each city rank customers by total completed spend.
WITH customer_spend AS (
    SELECT c.city,
           c.customer_id,
           SUM(o.order_amount) AS total_spend
    FROM orders    AS o
    JOIN customers AS c ON c.customer_id = o.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.city, c.customer_id
)
SELECT city,
       customer_id,
       ROUND(total_spend, 2) AS total_spend,
       RANK()       OVER (PARTITION BY city ORDER BY total_spend DESC) AS rnk,
       DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_rnk,
       ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS row_num
FROM customer_spend
ORDER BY city, row_num;

-- Q7: running total. Daily completed revenue (the day is the first ten characters of order_ts)
--     with a cumulative revenue column.
WITH daily AS (
    SELECT SUBSTR(order_ts, 1, 10)       AS day,
           ROUND(SUM(order_amount), 2)   AS revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY SUBSTR(order_ts, 1, 10)
)
SELECT day,
       revenue,
       ROUND(SUM(revenue) OVER (ORDER BY day), 2) AS cumulative_revenue
FROM daily
ORDER BY day;
