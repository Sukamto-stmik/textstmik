import csv
import psycopg2
conn = psycopg2.connect("host=localhost dbname=db_PDB user=postgres password=postgresql")
cur = conn.cursor()

#clean Dataset on table
cur.execute('truncate table public.tbl_dataset;')

#load Dataset
with open('D:\\Learning\\python\\project\\data_input\\data_input.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader) # Skip the header row.
    for row in reader:
        cur.execute(
        "INSERT INTO public.tbl_dataset VALUES (%s, %s, %s, %s, %s, %s)",
        row
    )
conn.commit()