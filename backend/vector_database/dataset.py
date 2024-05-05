import sqlite3


class Dataset:
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
        self.con = sqlite3.connect(db_path)

    def create_tables(self):
        """
        Creates the necessary tables in the database.
        """
        cur = self.con.cursor()
        cur.execute(
            """CREATE TABLE IF NOT EXISTS chunks(
                id INTEGER NOT NULL PRIMARY KEY, 
                title STRING NOT NULL, 
                text STRING NOT NULL
            )"""
        )

    def insert_chunks(self, chunks):
        """
        Inserts multiple chunks into the database.

        Args:
            chunks (list): A list of dictionaries representing the chunks to be inserted.
                Each dictionary should have the keys 'id', 'title', and 'text'.
        """
        data = [(chunk["id"], chunk["title"], chunk["text"]) for chunk in chunks]

        cur = self.con.cursor()
        cur.executemany("INSERT INTO chunks VALUES(?, ?, ?)", data)
        self.con.commit()

    def insert_chunk(self, chunk):
        """
        Inserts a single chunk into the database.

        Args:
            chunk (dict): A dictionary representing the chunk to be inserted.
                The dictionary should have the keys 'id', 'title', and 'text'.
        """
        self.insert_chunks([chunk])

    def search_chunk(self, id):
        """
        Searches for a chunk with the given ID in the database.

        Args:
            id (int): The ID of the chunk to search for.

        Returns:
            dict: A dictionary representing the found chunk, with the keys 'id', 'title', and 'text'.
        """
        cur = self.con.cursor()
        res = cur.execute("SELECT id, title, text FROM chunks WHERE id = ?", (id,))
        chunk = res.fetchone()

        return self._res_to_chunk(chunk)

    def search_chunks(self, ids):
        """
        Searches for multiple chunks with the given IDs in the database.

        Args:
            ids (list): A list of IDs of the chunks to search for.

        Returns:
            list: A list of dictionaries representing the found chunks, with the keys 'id', 'title', and 'text'.
        """
        cur = self.con.cursor()
        chunks = []

        # Split ids into chunks of 999
        for i in range(0, len(ids), 999):
            id_chunk = ids[i : i + 999]
            placeholders = ", ".join("?" for _ in id_chunk)
            query = f"SELECT id, title, text FROM chunks WHERE id IN ({placeholders})"
            res = cur.execute(query, id_chunk)
            chunks.extend(res.fetchall())

        return [self._res_to_chunk(chunk) for chunk in chunks]

    def _res_to_chunk(self, chunk):
        """
        Converts a result tuple from the database to a dictionary representing a chunk.

        Args:
            chunk (tuple): A tuple representing a chunk, with the elements 'id', 'title', and 'text'.

        Returns:
            dict: A dictionary representing the chunk, with the keys 'id', 'title', and 'text'.
        """
        return {"id": chunk[0], "title": chunk[1], "text": chunk[2]}


if __name__ == "__main__":
    dataset = Dataset(db_path="test.db")

    dataset.create_tables()

    chunks = [
        {"id": 1, "title": "First Chunk", "text": "This is the first chunk of data."},
        {"id": 2, "title": "Second Chunk", "text": "This is the second chunk of data."},
        {"id": 3, "title": "Third Chunk", "text": "This is the third chunk of data."},
        # Add more chunks as needed...
    ]

    dataset.insert_chunks(chunks)

    print(dataset.search_chunk(2))

    print(dataset.search_chunks([1, 3]))
