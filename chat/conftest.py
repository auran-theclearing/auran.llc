"""Compatibility shim for running tests in Python 3.10 sandbox.

Python 3.11 added datetime.UTC as an alias for datetime.timezone.utc.
The production server runs 3.11+, but the Cowork sandbox uses 3.10.
This conftest patches the datetime module so imports work in both.
"""

import datetime
import sys

if sys.version_info < (3, 11):
    datetime.UTC = datetime.timezone.utc
