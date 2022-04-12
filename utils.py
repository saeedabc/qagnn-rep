import time


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        m, s = divmod(int(te - ts), 60)
        print(f'Execution time: ({m}:{s}) min')

        return result
    return timed