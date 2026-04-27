"""Development entrypoint.

For production, use ``scripts/serve.sh`` (gunicorn). The Flask dev server
is intentionally NOT started in debug mode unless ``FLASK_DEBUG=1``.
"""
import os
from src.app import app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', '0').lower() in ('1', 'true', 'yes', 'on')
    host = os.environ.get('HOST', '127.0.0.1')
    app.run(host=host, port=port, debug=debug)
