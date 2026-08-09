import csv
import json
import hashlib
from collections import defaultdict
def main():
    with open('events.csv', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    valid_events = []
    seen_ids = set()
    for row in rows:
        try:
            points = int(row['points'])
        except ValueError:
            continue
        if row['id'] not in seen_ids:
            seen_ids.add(row['id'])
            valid_events.append(row)
    teams = defaultdict(int)
    for row in valid_events:
        teams[row['team']] += row['points']
    sorted_teams = sorted(teams.items())
    summary = {
        'teams': {k: v for k, v in sorted_teams},
        'valid_events': len(valid_events)
    }
    json_bytes = json.dumps(summary).encode('utf-8')
    checksum = hashlib.sha256(json_bytes).hexdigest()
    with open('summary.json', 'w') as f:
        f.write(json_bytes.decode('utf-8'))
    with open('checksum.txt', 'w') as f:
        f.write(checksum + '\n')
if __name__ == '__main__':
    main()