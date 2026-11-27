import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The shared app used by most tests must not rate limit them; the dedicated
# limiter tests build their own app with create_app().
os.environ.setdefault("RATE_LIMIT_ENABLED", "0")
