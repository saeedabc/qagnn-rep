import time


def timeit(func):
    """Decorator that reports the execution time."""

    def wrap(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        et = end - start

        h, et = divmod(et, 3600)
        m, s = divmod(et, 60)

        print(f'{func.__name__} execution time: [{int(h)}:{int(m)}:{int(s)}]')
        return result

    return wrap