#!/usr/bin/env python3
import json,os,sys
from urllib.request import Request,urlopen
port=os.getenv('RWKV_WORKER_PORT','18118')
req=Request(f'http://127.0.0.1:{port}/v1/statepool/drain',data=b'{"timeout_seconds":60}',headers={'Content-Type':'application/json'},method='POST')
try:
  with urlopen(req,timeout=10) as r: value=json.load(r)
  print(json.dumps(value,sort_keys=True))
  raise SystemExit(0 if value.get('status')=='safe_to_stop' else 1)
except Exception as exc:
  print(f'drain failed: {exc}',file=sys.stderr); raise SystemExit(1)
