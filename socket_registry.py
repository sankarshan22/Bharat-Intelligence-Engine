# ── socket_registry.py ──

_registry: dict = {}

def register(sid: str, socketio):
    _registry[sid] = socketio

def get(sid: str):
    return _registry.get(sid)

def unregister(sid: str):
    _registry.pop(sid, None)