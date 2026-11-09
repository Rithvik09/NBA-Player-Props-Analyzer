"""Read-only canonical export of logged forecasts; unknown provenance stays null.

Uses DATABASE_URL, never loads credentials from command-line arguments.
"""
import argparse
import json
from pathlib import Path
from src import db


FIELDS = {'player_id':'player_id', 'prop_type':'prop_type', 'model_version':'model_version',
          'game_id':'game_id', 'timestamp':'predicted_at', 'game_start':'game_start',
          'training_outcomes_through':'training_outcomes_through',
          'result_recorded_at':'result_recorded_at', 'predicted_value':'prediction',
          'actual_result':'actual', 'line':'line', 'over_probability':'over_probability',
          'feature_policy':'feature_policy', 'schedule_source':'schedule_source',
          'requested_at':'requested_at'}


def export(connection):
    connection.execute('SET TRANSACTION READ ONLY')
    columns = db.table_columns(connection, 'prediction_logs')
    if not columns:
        return []
    # Identifiers come exclusively from the fixed map above.
    selected = [key for key in FIELDS if key in columns]
    rows = connection.execute('SELECT ' + ', '.join('"' + key + '"' for key in selected) + ' FROM prediction_logs').fetchall()
    return [{FIELDS[key]: row[selected.index(key)] if key in selected else None
             for key in FIELDS} for row in rows]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with db.connect() as connection:
        result = export(connection)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str, allow_nan=False))
    print({'rows': len(result), 'output': str(args.output)})
