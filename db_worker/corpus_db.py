import psycopg2
from contextlib import contextmanager
from psycopg2 import pool
import logging


# Cannot use https://github.com/allenai/scholar-data-py/blob/master/pipeline/external_content/corpus_db/migration.py
# because it hardcodes the wrong user for dev environment. Fixing it properly is not trivial.
class CorpusDB:
    MAX_ATTEMPTS = 3
    '''
    NOTE: A prime candidate for removal if we decide to build
    a published Corpus HTTP client or access library.
    '''
    def __init__(
        self,
        dbname: str = 'corpus',
        host: str = 'localhost',
        port: int = 5432,
        user: str = None,
        password: str = None
    ):
        self.pool = pool.SimpleConnectionPool(
            1,
            1,
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port
        )

    @contextmanager
    def _get_cursor(self):
        try:
            with self.pool.getconn() as conn, conn.cursor() as cur:
                yield cur
        finally:
            self.pool.putconn(conn)

    def get_sourced_paper_pdf_hash(self, paper_id):
        logging.info('Paper ID %s', paper_id)
        for attempt_number in range(1, self.MAX_ATTEMPTS + 1):
            try:
                with self._get_cursor() as cur:
                    cur.execute('SELECT pdf_hash FROM sourced_paper WHERE id = %s' % paper_id)
                    row = cur.fetchone()
                    logging.info('Row %s', row)
                    return row[0] if row else None
            except psycopg2.DatabaseError:
                logging.exception('Exception when getting pdf hash for paper ID %s.', paper_id)
                if attempt_number == self.MAX_ATTEMPTS:
                    raise

    def close(self):
        self.pool.closeall()
