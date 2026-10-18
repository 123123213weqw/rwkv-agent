# Event summary
Input CSV columns: id,team,points. Ignore rows with a non-integer points value. For duplicate ids keep only the first valid occurrence. Aggregate points by team, sort team keys alphabetically, and emit compact JSON with keys teams then valid_events. checksum.txt is the SHA-256 of the exact summary.json bytes followed by a newline.
