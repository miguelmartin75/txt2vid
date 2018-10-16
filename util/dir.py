import os

def ensure_exists(directories):
    try:
        import os
        os.makedirs(directories)
    except OSError:
        pass
