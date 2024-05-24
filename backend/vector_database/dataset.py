import sqlite3
import os
from typing import Iterable


class DatasetSQL:
    """
    Represents a dataset that interacts with a SQLite database.

    Args:
        db_path (str): The path to the SQLite database file.

    Attributes:
        con (sqlite3.Connection): The connection to the SQLite database.

    Methods:
        create_tables(): Creates the necessary tables in the database.
        insert_chunks(chunks): Inserts multiple chunks into the database.
        insert_chunk(chunk): Inserts a single chunk into the database.
        search_chunk(id): Searches for a chunk with the given ID in the database.
        search_chunks(ids): Searches for multiple chunks with the given IDs in the database.
    """

    def __init__(self, *, db_path):
        self.db_path = db_path

    def drop_tables(self):
        """
        Drops the all the tables in the database.

        Returns:
            None
        """
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        cur.execute("DROP TABLE IF EXISTS chunks")
        con.close()

    def create_tables(self):
        """
        Creates the necessary tables in the database.
        """
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        cur.execute(
            """CREATE TABLE IF NOT EXISTS chunks(
                id INTEGER NOT NULL PRIMARY KEY, 
                titles STRING NOT NULL, 
                text STRING NOT NULL
            )"""
        )
        con.close()

    def insert_chunks(self, chunks: Iterable[dict]):
        """
        Inserts multiple chunks into the database.

        Args:
            chunks (list): A list of dictionaries representing the chunks to be inserted.
                Each dictionary should have the keys 'id', 'titles', and 'text'.
        """
        data = [(chunk["id"], chunk["titles"], chunk["text"]) for chunk in chunks]

        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        cur.executemany("INSERT INTO chunks VALUES(?, ?, ?)", data)
        con.commit()
        con.close()

    def insert_chunk(self, chunk):
        """
        Inserts a single chunk into the database.

        Args:
            chunk (dict): A dictionary representing the chunk to be inserted.
                The dictionary should have the keys 'id', 'titles', and 'text'.
        """
        self.insert_chunks([chunk])

    def search_chunk(self, id):
        """
        Searches for a chunk with the given ID in the database.

        Args:
            id (int): The ID of the chunk to search for.

        Returns:
            dict: A dictionary representing the found chunk, with the keys 'id', 'titles', and 'text'.
        """
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        res = cur.execute(
            "SELECT id, titles, text FROM chunks WHERE id = ?", (int(id),)
        )
        chunk = res.fetchone()
        con.close()
        return self._res_to_chunk(chunk)

    def search_chunks(self, ids):
        """
        todo: test
        Searches for multiple chunks with the given IDs in the database.

        Args:
            ids (list): A list of IDs of the chunks to search for.

        Returns:
            list: A list of dictionaries representing the found chunks, with the keys 'id', 'titles', and 'text'.
        """
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        chunks = []

        # Split ids into chunks of 999
        for i in range(0, len(ids), 999):
            id_chunk = ids[i : i + 999]
            placeholders = ", ".join("?" for _ in id_chunk)
            query = f"""
                SELECT id, titles, text FROM chunks 
                WHERE id IN ({placeholders}) ORDER BY id
            """
            res = cur.execute(query, id_chunk)
            chunks.extend(res.fetchall())

        con.close()

        return [self._res_to_chunk(chunk) for chunk in chunks]

    def paginate_chunks(self, count_per_page, offset=0) -> Iterable[list]:
        """
        Iterate over the chunks in the database, paginating them.
        :param count_per_page: Page size
        :param offset: Offset
        """

        con = sqlite3.connect(self.db_path)
        while True:
            cur = con.cursor()
            res = cur.execute(
                """
                    SELECT id, titles, text FROM chunks ORDER BY id 
                    LIMIT ? OFFSET ? 
                """,
                (count_per_page, offset),
            )

            chunks = [self._res_to_chunk(chunk) for chunk in res.fetchall()]

            if len(chunks) > 0:
                offset += len(chunks)
                yield chunks
            else:
                break

        con.close()

    def count_of_chunks(self):
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        res = cur.execute("SELECT COUNT(*) FROM chunks")
        count = res.fetchone()[0]
        con.close()
        return count

    def _res_to_chunk(self, chunk):
        """
        Converts a result tuple from the database to a dictionary representing a chunk.

        Args:
            chunk (tuple): A tuple representing a chunk, with the elements 'id', 'titles', and 'text'.

        Returns:
            dict: A dictionary representing the chunk, with the keys 'id', 'titles', and 'text'.
        """
        return {"id": chunk[0], "titles": chunk[1], "text": chunk[2]}


class MockDataset:
    """A mock dataset class for testing"""

    def __init__(self, chunks: list):
        self.chunks = chunks
        self.titles = ["title" for _ in chunks]

    def search_chunk(self, id) -> str:
        return {
            "text": self.chunks[id],
            "titles": self.titles[id],
        }

    def search_chunks(self, ids: list):
        return [chunk for index, chunk in enumerate(self.chunks) if index in ids]


DB_DIR_PATH = "backend/vector_database/data"

if __name__ == "__main__":
    if not os.path.exists(DB_DIR_PATH):
        os.mkdir(DB_DIR_PATH)

    dataset = DatasetSQL(db_path=DB_DIR_PATH + "/test.db")

    dataset.create_tables()

    chunks = [
        {"id": 1, "titles": "First Chunk", "text": "This is the first chunk of data."},
        {
            "id": 2,
            "titles": "Second Chunk",
            "text": "This is the second chunk of data.",
        },
        {"id": 3, "titles": "Third Chunk", "text": "This is the third chunk of data."},
        # Add more chunks as needed...
    ]

    dataset.insert_chunks(chunks)

    print(dataset.search_chunk(2))

    print(dataset.search_chunks([1, 3]))
