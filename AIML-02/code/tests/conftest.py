import sys
from pathlib import Path

aiml02_code = Path(__file__).resolve().parent.parent
if str(aiml02_code) not in sys.path:
    sys.path.insert(0, str(aiml02_code))

for mod in list(sys.modules.keys()):
    if mod == "src" or mod.startswith("src."):
        del sys.modules[mod]
