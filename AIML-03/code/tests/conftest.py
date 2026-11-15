import sys
from pathlib import Path

# Ensure AIML-03 code directory is prioritized when running AIML-03 tests
aiml03_code = Path(__file__).resolve().parent.parent
if str(aiml03_code) not in sys.path:
    sys.path.insert(0, str(aiml03_code))

# Remove cached 'src' modules from AIML-02 if present
for mod in list(sys.modules.keys()):
    if mod == "src" or mod.startswith("src."):
        del sys.modules[mod]
