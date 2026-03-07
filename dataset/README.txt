To load the data use the data following code:

import json
import gzip

with gzip.open("XXXX.json.gz", "rt") as f:
        dict = json.load(f)

