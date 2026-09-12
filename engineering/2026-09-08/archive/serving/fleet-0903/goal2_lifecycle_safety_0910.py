"""Recovery helpers for goal2 controllers; never signal any process themselves."""
import json
import time


def settle_termination(exact, stopped):
    """After SIGTERM was sent, cancellation must still await its actual exit."""
    if stopped or not exact.termination_sent:
        return stopped
    report = 0.0
    while not exact.exited():
        now = time.monotonic()
        if now - report >= 30:
            print(json.dumps(dict(waiting_for_selected_termination=True)), flush=True)
            report = now
        time.sleep(.1)
    return True


def wait_peer_idle(guard):
    """Wait through peer work during restoration; identity failures stay fatal."""
    report = 0.0
    while True:
        state, _ = guard.inspect()
        if not state['busy']:
            return
        now = time.monotonic()
        if now - report >= 30:
            print(json.dumps(dict(waiting_for_peer_before_restoration=True)), flush=True)
            report = now
        time.sleep(.5)


def wait_endpoint_idle(health, timeout=120):
    """An HTTP body can arrive just before Handler.finally releases its lock."""
    deadline = time.monotonic() + timeout
    while health()['busy']:
        if time.monotonic() >= deadline:
            raise RuntimeError('Endpoint remained busy after completed validation')
        time.sleep(.05)
