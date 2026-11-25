"""Application signal coordination; model ownership stays on one joined worker."""
import signal
import threading


def run_owned(work):
    """Defer process termination until native calls and explicit cleanup finish."""
    stopping = threading.Event()
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    errors, results = [], []

    def stop(*_):
        stopping.set()

    def own():
        # Native runtime children inherit this mask. The main thread receives
        # termination signals without interrupting model creation or RPC calls.
        signal.pthread_sigmask(signal.SIG_BLOCK, previous)
        try:
            results.append(work(stopping))
        except BaseException as exc:
            errors.append(exc)

    for sig in previous:
        signal.signal(sig, stop)
    worker = threading.Thread(target=own, name='audio8-owner', daemon=False)
    try:
        worker.start()
        while worker.is_alive():
            worker.join(.1)
        if errors:
            raise errors[0]
        return results[0]
    finally:
        stopping.set()
        if worker.ident is not None:
            worker.join()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
