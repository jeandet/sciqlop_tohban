def load(main_window):
    # Imported lazily so the pure .evt/timeline modules stay usable without SciQLop.
    from .plugin import load as _load

    return _load(main_window)
