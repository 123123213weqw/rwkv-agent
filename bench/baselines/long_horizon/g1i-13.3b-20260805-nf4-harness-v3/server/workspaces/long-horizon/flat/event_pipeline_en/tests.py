import json
import subprocess
from pathlib import Path
subprocess.run(['python3','summarize.py','events.csv','summary.json','checksum.txt'], check=True)
raw = Path('summary.json').read_bytes()
assert raw == b'{"teams":{"blue":7,"green":4,"red":8},"valid_events":4}\n', raw
assert Path('checksum.txt').read_text() == '59aa3c6a6f1cb787297bc44aca7a91ee8e1918f07dec96a3405d76698b98c83d\n'
print('PIPELINE_OK')
