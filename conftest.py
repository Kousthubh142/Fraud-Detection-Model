import sys
from pathlib import Path

# Add src/ to path so all tests can import from it
sys.path.insert(0, str(Path(__file__).parent / "src"))