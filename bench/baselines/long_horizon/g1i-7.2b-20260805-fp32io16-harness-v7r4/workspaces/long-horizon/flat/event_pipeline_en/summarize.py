import csv
import json
import hashlib
import sys
def main():
    csv_path = 'events.csv'
    json_path = 'summary.json'
    checksum_path = 'checksum.txt'
    teams = {}
    valid_events = 0
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                points = int(row['points'])
            except ValueError:
                continue
            team = row['team']
            if team not in teams:
                teams[team] = 0
            teams[team] += points
            valid_events += 1
    sorted_teams = sorted(teams.items(), key=lambda x: x[0])
    json_data = {
        'teams': {k: v for k, v in sorted_teams},
        'valid_events': valid_events
    }
    json_bytes = json.dumps(json_data).encode('utf-8')
    checksum = hashlib.sha256(json_bytes).hexdigest()
    with open(json_path, 'w') as f:
        f.write(json_bytes.decode('utf-8'))
    with open(checksum_path, 'w') as f:
        f.write(checksum + '\n')
    print('PIPELINE_OK')
if __name__ == '__main__':
    main()