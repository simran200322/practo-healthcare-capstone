"""Practo Healthcare Capstone - Domain Support Agent package."""
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

__version__ = "1.0.0"
