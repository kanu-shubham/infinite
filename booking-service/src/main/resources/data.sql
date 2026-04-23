-- Seed data: one movie, one show, a small seat grid.
-- With ddl-auto=create-drop this runs after Hibernate creates the schema.
INSERT INTO movie (id, title, genre, duration_minutes) VALUES (1, 'Inception', 'Sci-Fi', 148);
INSERT INTO movie (id, title, genre, duration_minutes) VALUES (2, 'The Dark Knight', 'Action', 152);

INSERT INTO show (id, movie_id, screen, start_time, total_seats) VALUES (1, 1, 'Screen 1', '2026-05-01 19:00:00', 20);
INSERT INTO show (id, movie_id, screen, start_time, total_seats) VALUES (2, 2, 'Screen 2', '2026-05-01 21:00:00', 20);

-- 20 seats per show, labelled A1..A10, B1..B10
-- (We avoid a procedural block so H2 and Postgres both accept it.)
INSERT INTO seat (id, show_id, seat_label, status, version) VALUES
  (1, 1, 'A1', 'AVAILABLE', 0), (2, 1, 'A2', 'AVAILABLE', 0),
  (3, 1, 'A3', 'AVAILABLE', 0), (4, 1, 'A4', 'AVAILABLE', 0),
  (5, 1, 'A5', 'AVAILABLE', 0), (6, 1, 'A6', 'AVAILABLE', 0),
  (7, 1, 'A7', 'AVAILABLE', 0), (8, 1, 'A8', 'AVAILABLE', 0),
  (9, 1, 'A9', 'AVAILABLE', 0), (10, 1, 'A10', 'AVAILABLE', 0),
  (11, 1, 'B1', 'AVAILABLE', 0), (12, 1, 'B2', 'AVAILABLE', 0),
  (13, 1, 'B3', 'AVAILABLE', 0), (14, 1, 'B4', 'AVAILABLE', 0),
  (15, 1, 'B5', 'AVAILABLE', 0), (16, 1, 'B6', 'AVAILABLE', 0),
  (17, 1, 'B7', 'AVAILABLE', 0), (18, 1, 'B8', 'AVAILABLE', 0),
  (19, 1, 'B9', 'AVAILABLE', 0), (20, 1, 'B10', 'AVAILABLE', 0),
  (21, 2, 'A1', 'AVAILABLE', 0), (22, 2, 'A2', 'AVAILABLE', 0),
  (23, 2, 'A3', 'AVAILABLE', 0), (24, 2, 'A4', 'AVAILABLE', 0),
  (25, 2, 'A5', 'AVAILABLE', 0), (26, 2, 'A6', 'AVAILABLE', 0),
  (27, 2, 'A7', 'AVAILABLE', 0), (28, 2, 'A8', 'AVAILABLE', 0),
  (29, 2, 'A9', 'AVAILABLE', 0), (30, 2, 'A10', 'AVAILABLE', 0),
  (31, 2, 'B1', 'AVAILABLE', 0), (32, 2, 'B2', 'AVAILABLE', 0),
  (33, 2, 'B3', 'AVAILABLE', 0), (34, 2, 'B4', 'AVAILABLE', 0),
  (35, 2, 'B5', 'AVAILABLE', 0), (36, 2, 'B6', 'AVAILABLE', 0),
  (37, 2, 'B7', 'AVAILABLE', 0), (38, 2, 'B8', 'AVAILABLE', 0),
  (39, 2, 'B9', 'AVAILABLE', 0), (40, 2, 'B10', 'AVAILABLE', 0);
