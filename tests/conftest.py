import os
import sys

# make the project root importable when pytest is run from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
