#!/usr/bin/env python3
import json
import sys

def migrate_config(input_path, output_path):
    with open(input_path, 'r') as f:
        data = json.load(f)

    # Rename timeout_ms to timeout_seconds using exact integer division by 1000
    if 'timeout_ms' in data:
        data['timeout_seconds'] = data['timeout_ms'] // 1000
        del data['timeout_ms']

    # Rename debug to log_level: debug=true becomes debug, false becomes info
    if 'debug' in data:
        if data['debug']:
            data['log_level'] = 'debug'
        else:
            data['log_level'] = 'info'
        del data['debug']

    # Preserve name
    if 'name' in data:
        data['name'] = data['name']

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

def main():
    if len(sys.argv) != 3:
        print("Usage: migrate.py <input> <output>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    migrate_config(input_path, output_path)

if __name__ == '__main__':
    main()