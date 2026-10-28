"""Import a local policy directory without reseeding business data."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.backend.db import get_engine, get_session_factory, init_db
from src.policy import DEFAULT_POLICY_DIR, import_policy_documents


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', nargs='?', default=os.environ.get('SUPPORT_POLICY_DIR') or DEFAULT_POLICY_DIR)
    args = parser.parse_args()
    engine = get_engine()
    try:
        init_db(engine)
        print(json.dumps(import_policy_documents(args.directory, get_session_factory(engine)), ensure_ascii=False))
    finally:
        engine.dispose()
