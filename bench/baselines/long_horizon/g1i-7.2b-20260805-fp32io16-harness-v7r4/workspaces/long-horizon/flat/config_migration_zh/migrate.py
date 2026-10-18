import json
from pathlib import Path
expected = {
    'service.json': {'name': 'api', 'timeout_seconds': 5, 'log_level': 'debug', 'schema_version': 2},
    'worker.json': {'name': 'jobs', 'timeout_seconds': 12, 'log_level': 'info', 'schema_version': 2},
}
for name, value in expected.items():
    assert json.loads(Path(name).read_text()) == value, (name, json.loads(Path(name).read_text()))
report = json.loads(Path('migration-report.json').read_text())
assert report == {'migrated': ['service.json', 'worker.json'], 'schema_version': 2}
print('MIGRATION_OK')