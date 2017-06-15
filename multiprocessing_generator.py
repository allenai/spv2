# -*- encoding: utf-8 -*-
import queue
from multiprocessing import Process, Queue


class ExceptionItem(object):
    def __init__(self, exception):
        self.exception = exception


class ParallelGeneratorException(Exception):
    pass


class GeneratorDied(ParallelGeneratorException):
    pass


class ParallelGenerator(object):
    def __init__(self, orig_gen, max_lookahead=None, get_timeout=10):
        """
        Creates a parallel generator from a normal one.
        The elements will be prefetched up to max_lookahead
        ahead of the consumer. If max_lookahead is None,
        everything will be fetched.

        The get_timeout parameter is the number of seconds
        after which we check that the subprocess is still
        alive, when waiting for an element to be generated.

        Any exception raised in the generator will
        be forwarded to this parallel generator.
        """
        if max_lookahead:
            self.queue = Queue(max_lookahead)
        else:
            self.queue = Queue()

        def wrapped():
            try:
                for item in orig_gen:
                    self.queue.put(item)
                raise StopIteration()
            except Exception as e:
                self.queue.put(ExceptionItem(e))

        self.get_timeout = get_timeout

        self.process = Process(target=wrapped)
        self.process_started = False

    def __enter__(self):
        """
        Starts the process
        """
        self.process.start()
        self.process_started = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Kills the process
        """
        if self.process and self.process.is_alive():
            self.process.terminate()

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self

    def next(self):
        if not self.process_started:
            raise ParallelGeneratorException(
                """The generator has not been started.
                   Please use "with ParallelGenerator(..) as g:"
                """
            )
        try:
            item_received = False
            while not item_received:
                try:
                    item = self.queue.get(timeout=self.get_timeout)
                    item_received = True
                except queue.Empty:
                    # check that the process is still alive
                    if not self.process.is_alive():
                        raise GeneratorDied("The generator died unexpectedly.")

            if type(item) == ExceptionItem:
                raise item.exception
            return item

        except Exception:
            self.queue = None
            if self.process.is_alive():
                self.process.terminate()
            self.process = None
            raise
