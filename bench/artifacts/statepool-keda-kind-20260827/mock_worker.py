#!/usr/bin/env python3
from __future__ import annotations
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json, os, threading, time
from urllib.request import Request, urlopen

statepool=os.environ['RWKV_STATEPOOL_URL'].rstrip('/')
worker_id=(os.getenv('RWKV_WORKER_ID') or os.getenv('POD_UID') or os.getenv('POD_NAME') or 'mock-worker')
port=int(os.getenv('RWKV_WORKER_PORT','18118'))
endpoint=os.getenv('RWKV_WORKER_ENDPOINT',f'http://{os.getenv("POD_IP","127.0.0.1")}:{port}')
delay=float(os.getenv('REGISTER_DELAY_SECONDS','15'))
lifecycle='starting'
lock=threading.Lock()

def post(path,payload,timeout=5):
    req=Request(statepool+path,data=json.dumps(payload,separators=(',',':')).encode(),headers={'Content-Type':'application/json'},method='POST')
    with urlopen(req,timeout=timeout) as response:
        raw=response.read()
        return json.loads(raw) if raw else {'status':'ok'}

def capability():
    with lock: current=lifecycle
    return {
      'contract_version':'statepool-worker-capability.v1',
      'worker_id':worker_id,'zone':os.getenv('RWKV_WORKER_ZONE','cloud'),
      'endpoint':endpoint,'lifecycle':current,
      'models':[{
        'model_id':os.environ['RWKV_WORKER_MODEL_ID'],
        'revision':os.environ['RWKV_WORKER_MODEL_REVISION'],
        'tokenizer':os.environ['RWKV_WORKER_TOKENIZER'],
        'state_abi':os.environ['RWKV_WORKER_STATE_ABI']}],
      'device':{'vendor':os.getenv('RWKV_WORKER_DEVICE_VENDOR','simulated'),
        'model':os.getenv('RWKV_WORKER_DEVICE_MODEL','kind-mock'),
        'runtime':os.getenv('RWKV_WORKER_DEVICE_RUNTIME','none'),
        'memory_bytes':int(os.getenv('RWKV_WORKER_DEVICE_MEMORY_BYTES','0'))},
      'capacity':{'state_slots':int(os.getenv('RWKV_WORKER_STATE_SLOTS','8')),
        'free_state_slots':int(os.getenv('RWKV_WORKER_STATE_SLOTS','8')),
        'max_batch':int(os.getenv('RWKV_WORKER_MAX_BATCH','8')),
        'queue_depth':0,'running_requests':0,'unpersisted_state_slots':0},
      'price':{'currency':os.getenv('RWKV_WORKER_PRICE_CURRENCY','CNY'),
        'per_gpu_hour':float(os.getenv('RWKV_WORKER_PRICE_PER_GPU_HOUR','0'))},
      'labels':{'test_mode':'keda-simulation'},'reported_at_ms':int(time.time()*1000)}

def heartbeat_loop():
    global lifecycle
    time.sleep(delay)
    with lock: lifecycle='ready'
    registered=False
    while True:
      try:
        payload=capability()
        path=f'/plugin/v1/workers/{worker_id}/heartbeat' if registered else '/plugin/v1/workers/register'
        response=post(path,payload)
        registered=True
        print(json.dumps({'event':'heartbeat','worker_id':worker_id,'lifecycle':payload['lifecycle'],'response':response},sort_keys=True),flush=True)
      except Exception as exc:
        registered=False
        print(json.dumps({'event':'heartbeat_error','worker_id':worker_id,'error':str(exc)},sort_keys=True),flush=True)
      time.sleep(2)

class Handler(BaseHTTPRequestHandler):
    def log_message(self,fmt,*args): return
    def send_json(self,status,value):
      body=json.dumps(value,separators=(',',':')).encode(); self.send_response(status)
      self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
      if self.path=='/live': self.send_json(200,{'status':'live'}); return
      if self.path=='/ready':
        with lock: current=lifecycle
        self.send_json(503 if current=='draining' else 200,{'status':current}); return
      if self.path=='/health': self.send_json(200,{'status':'ready','model_ref':capability()['models'][0]}); return
      if self.path=='/v1/statepool/drain': self.send_json(200,{'contract_version':'statepool-drain-status.v1','worker_id':worker_id,'status':'safe_to_stop','active_requests':0,'unpersisted_states':0}); return
      self.send_json(404,{'error':'not_found'})
    def do_POST(self):
      global lifecycle
      if self.path!='/v1/statepool/drain': self.send_json(404,{'error':'not_found'}); return
      length=int(self.headers.get('Content-Length','0')); self.rfile.read(length)
      with lock: lifecycle='draining'
      payload=capability()
      try: post(f'/plugin/v1/workers/{worker_id}/heartbeat',payload)
      except Exception as exc: print(json.dumps({'event':'drain_heartbeat_error','error':str(exc)}),flush=True)
      try:
        remote=post(f'/plugin/v1/workers/{worker_id}/drain',{'contract_version':'statepool-drain-request.v1','deadline_ms':int((time.time()+60)*1000)})
      except Exception as exc: remote={'status':'control_plane_unavailable','error':str(exc)}
      result={'contract_version':'statepool-drain-status.v1','worker_id':worker_id,'status':'safe_to_stop','active_requests':0,'unpersisted_states':0,'control_plane':remote}
      print(json.dumps({'event':'prestop_safe_to_stop','result':result},sort_keys=True),flush=True)
      self.send_json(200,result)

threading.Thread(target=heartbeat_loop,daemon=True).start()
print(json.dumps({'event':'server_started','worker_id':worker_id,'register_delay_seconds':delay},sort_keys=True),flush=True)
ThreadingHTTPServer(('0.0.0.0',port),Handler).serve_forever()
