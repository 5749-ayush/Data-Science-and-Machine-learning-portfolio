import sqlite3
import pandas as pd
import os

# Define the name of your SQLite database file
DB_NAME = 'chinook_music.db'

# Remove existing database file if it exists to start fresh
if os.path.exists(DB_NAME):
    os.remove(DB_NAME)
    print(f"Removed existing database: {DB_NAME}")

# Connect to SQLite database (it will create the file if it doesn't exist)
conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()

print(f"Connected to database: {DB_NAME}")

# Define table creation SQL statements based on your schema
# Order matters due to foreign key constraints!
table_creation_sqls = {
    'employee': """
        CREATE TABLE IF NOT EXISTS employee (
            employee_id INTEGER PRIMARY KEY,
            last_name TEXT,
            first_name TEXT,
            title TEXT,
            reports_to INTEGER,
            levels TEXT,
            birthdate TEXT, -- Assuming date as text for simplicity
            hire_date TEXT, -- Assuming date as text for simplicity
            address TEXT,
            city TEXT,
            state TEXT,
            country TEXT,
            postal_code TEXT,
            phone TEXT,
            fax TEXT,
            email TEXT,
            FOREIGN KEY (reports_to) REFERENCES employee (employee_id)
        );
    """,
    'customer': """
        CREATE TABLE IF NOT EXISTS customer (
            customer_id INTEGER PRIMARY KEY,
            first_name TEXT,
            last_name TEXT,
            company TEXT,
            address TEXT,
            city TEXT,
            state TEXT,
            country TEXT,
            postal_code TEXT,
            phone TEXT,
            fax TEXT,
            email TEXT,
            support_rep_id INTEGER,
            FOREIGN KEY (support_rep_id) REFERENCES employee (employee_id)
        );
    """,
    'genre': """
        CREATE TABLE IF NOT EXISTS genre (
            genre_id INTEGER PRIMARY KEY,
            name TEXT
        );
    """,
    'media_type': """
        CREATE TABLE IF NOT EXISTS media_type (
            media_type_id INTEGER PRIMARY KEY,
            name TEXT
        );
    """,
    'artist': """
        CREATE TABLE IF NOT EXISTS artist (
            artist_id INTEGER PRIMARY KEY,
            name TEXT
        );
    """,
    'album': """
        CREATE TABLE IF NOT EXISTS album (
            album_id INTEGER PRIMARY KEY,
            title TEXT,
            artist_id INTEGER,
            FOREIGN KEY (artist_id) REFERENCES artist (artist_id)
        );
    """,
    'track': """
        CREATE TABLE IF NOT EXISTS track (
            track_id INTEGER PRIMARY KEY,
            name TEXT,
            album_id INTEGER,
            media_type_id INTEGER,
            genre_id INTEGER,
            composer TEXT,
            milliseconds INTEGER,
            bytes INTEGER,
            unit_price REAL,
            FOREIGN KEY (album_id) REFERENCES album (album_id),
            FOREIGN KEY (media_type_id) REFERENCES media_type (media_type_id),
            FOREIGN KEY (genre_id) REFERENCES genre (genre_id)
        );
    """,
    'invoice': """
        CREATE TABLE IF NOT EXISTS invoice (
            invoice_id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            invoice_date TEXT, -- Assuming date as text for simplicity
            billing_address TEXT,
            billing_city TEXT,
            billing_state TEXT,
            billing_country TEXT,
            billing_postal_code TEXT,
            total REAL,
            FOREIGN KEY (customer_id) REFERENCES customer (customer_id)
        );
    """,
    'invoice_line': """
        CREATE TABLE IF NOT EXISTS invoice_line (
            invoice_line_id INTEGER PRIMARY KEY,
            invoice_id INTEGER,
            track_id INTEGER,
            unit_price REAL,
            quantity INTEGER,
            FOREIGN KEY (invoice_id) REFERENCES invoice (invoice_id),
            FOREIGN KEY (track_id) REFERENCES track (track_id)
        );
    """,
    'playlist': """
        CREATE TABLE IF NOT EXISTS playlist (
            playlist_id INTEGER PRIMARY KEY,
            name TEXT
        );
    """,
    'playlist_track': """
        CREATE TABLE IF NOT EXISTS playlist_track (
            playlist_id INTEGER,
            track_id INTEGER,
            PRIMARY KEY (playlist_id, track_id), -- Composite primary key
            FOREIGN KEY (playlist_id) REFERENCES playlist (playlist_id),
            FOREIGN KEY (track_id) REFERENCES track (track_id)
        );
    """
}

# CSV file names and their corresponding table names
# Ensure these match your actual CSV file names exactly (case-sensitive)
csv_files = {
    'employee.csv': 'employee',
    'customer.csv': 'customer',
    'genre.csv': 'genre',
    'media_type.csv': 'media_type',
    'artist.csv': 'artist',
    'album.csv': 'album', # Ensure you use the correct album.csv if you have multiple
    'track.csv': 'track',
    'invoice.csv': 'invoice',
    'invoice_line.csv': 'invoice_line',
    'playlist.csv': 'playlist',
    'playlist_track.csv': 'playlist_track',
    'alumbs2.csv': 'alumb'
    }

# Execute table creation SQLs in order
# Tables with foreign keys must be created AFTER the tables they reference.
ordered_tables = [
    'employee', 'customer', 'genre', 'media_type', 'artist',
    'album', 'track', 'invoice', 'invoice_line', 'playlist', 'playlist_track'
]

for table_name in ordered_tables:
    sql = table_creation_sqls[table_name]
    try:
        cursor.execute(sql)
        print(f"Table '{table_name}' created successfully.")
    except sqlite3.Error as e:
        print(f"Error creating table '{table_name}': {e}")
        conn.close()
        exit() # Exit if a critical table cannot be created

# Import data from CSVs into corresponding tables
for csv_file, table_name in csv_files.items():
    try:
        df = pd.read_csv(csv_file)
        # Clean column names to be valid SQL identifiers (lowercase, no spaces/special chars)
        df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('[^a-z0-9_]', '', regex=True)

        # Handle specific column name discrepancies if necessary
        # Example: if your album.csv had 'album_id' but you wanted 'AlbumId' in SQL
        # df.rename(columns={'album_id': 'AlbumId'}, inplace=True)

        df.to_sql(table_name, conn, if_exists='append', index=False)
        print(f"Data from '{csv_file}' imported into '{table_name}' successfully.")
    except FileNotFoundError:
        print(f"Error: '{csv_file}' not found. Skipping data import for this table.")
    except Exception as e:
        print(f"Error importing data from '{csv_file}' into '{table_name}': {e}")

# Commit changes and close the connection
conn.commit()
conn.close()
print("Database creation and data import complete.")