"""Run with python -m scripts.ingest_knowledge --root knowledge."""
import argparse
import json
from src.knowledge import KnowledgeStore


def main():
    parser = argparse.ArgumentParser(description='离线导入单平台知识资料；请先停止应用')
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    print(json.dumps(KnowledgeStore().ingest(args.root),ensure_ascii=False))


if __name__ == '__main__':
    main()
