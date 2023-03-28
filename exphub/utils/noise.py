import os
import contextlib


class Suppressor:

    @classmethod
    def exec_no_stdout(cls, fn, **kwargs):
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                return fn(**kwargs)
