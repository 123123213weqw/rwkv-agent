import json
from pathlib import Path
def migrate_config(input_path, output_path):
    with open(input_path, 'r') as f:
        data = json.load(f)
    # Convert timeout_ms to timeout_seconds using exact integer division by 1000
    if 'timeout_ms' in data:
        data['timeout_seconds'] = data['timeout_ms'] // 1000
        del data['timeout_ms']
    # Convert debug to log_level
    if 'debug' in data:
        data['log_level'] = 'debug' if data['debug'] else 'info'
        del data['debug']
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
def main():
    migrate_config('service.json', 'service.json')
    migrate_config('worker.json', 'worker.json')
if __name__ == '__main__':
    main()