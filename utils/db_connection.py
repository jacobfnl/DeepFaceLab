import os
import mysql.connector


def open_db_connection():
    DB_USER = os.getenv('DB_USER')
    DB_PASS = os.getenv('DB_PASS')
    DB_ADDRESS = os.getenv('DB_ADDRESS')
    DB_DB = os.getenv('DB_DB')
    config = {
        'user': DB_USER,
        'password': DB_PASS,
        'host': DB_ADDRESS,
        'database': DB_DB,
        'raise_on_warnings': True
    }
    return mysql.connector.connect(**config)