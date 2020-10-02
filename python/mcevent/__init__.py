"""Multi-consumer multi-producer dispatching mechanism

Originally based on pydispatch (BSD) https://pypi.org/project/PyDispatcher/2.0.1/
See license.txt for original license.

Heavily modified for Django's purposes.
"""

from .dispatcher import Signal, receiver  # NOQA

dispatch = Signal()  # NOQA


class SocketIOMock:
    mock = True

    def on(self, event, **kwargs):
        def _decorator(func):
            return func

        return _decorator

    def emit(*args, **kwargs):
        pass


sio = SocketIOMock()  # NOQA
