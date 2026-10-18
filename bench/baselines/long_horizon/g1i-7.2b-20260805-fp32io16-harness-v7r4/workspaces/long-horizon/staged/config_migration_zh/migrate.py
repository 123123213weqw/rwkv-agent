import json
from pathlib import Path
def migrate_config(input_path, output_path):
    with open(input_path, 'r') as f:
        data = json.load(f)
    # Rename timeout_ms to timeout_seconds using exact integer division by 1000
    data['timeout_seconds'] = data.pop('timeout_ms') // 1000
    # Rename debug to log_level: debug=true becomes debug, false becomes info
    if data.get('log_level') == 'debug':
        data['log_level'] = 'debug'
    elif data.get('log_level') == 'false':
        data['log_level'] = 'info'
    # Preserve name
    data['name'] = data.pop('name')
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
def main():
    migrate_config('service.json', 'service.json')
    migrate_config('worker.json', 'worker.json')
    # Verify migration
    assert json.loads(Path('service.json').read_text()) == {
        'name': 'api',
        'timeout_seconds': 5,
        'log_level': 'debug',
        'schema_version': 2
    }
    assert json.loads(Path('worker.json').read_text()) == {
        'name': 'jobs',
        'timeout_seconds': 12,
        'log_level': 'info',
        'schema_version': 2
    }
    print('MIGRATION_OK')
if __name__ == '__main__':
    main()