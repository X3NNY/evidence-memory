"""Synthetic TCP checks inside the built image; Docker networking is disabled."""
import argparse
from contextlib import closing
import json
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))

KEY = 'synthetic-deployment-check-only'


def server(database, port, restart):
    import numpy as np
    import uvicorn
    from memory_bench.service.app import Settings, create_app

    class Fixture:
        def encode(self, texts, **kwargs):
            if restart:
                raise RuntimeError('restart must reuse durable vectors')
            result = []
            for text in texts:
                if 'physician' in text or text == 'doctor':
                    result.append([1., 0., 0., 0.])
                elif 'bakery' in text or text == 'bread':
                    result.append([0., 1., 0., 0.])
                else:
                    raise RuntimeError('uncached synthetic fixture')
            return np.array(result)

    app = create_app(Settings(database, KEY), embedder=Fixture(),
                     embedding_namespace='deployment-synthetic-only-v1', embedding_dimension=4)
    uvicorn.run(app, host='127.0.0.1', port=port, access_log=False, log_level='error')


def run():
    checks, processes = [], []
    with tempfile.TemporaryDirectory(prefix='agent-memory-check-') as directory:
        database = Path(directory) / 'memory.sqlite3'
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        base = f'http://127.0.0.1:{port}'

        def call(path, payload=None, key=KEY):
            data = None if payload is None else json.dumps(payload).encode()
            headers = {'Content-Type': 'application/json'}
            if key is not None:
                headers['Authorization'] = 'Bearer ' + key
            request = urllib.request.Request(base + path, data=data, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=10) as response:
                    return response.status, json.load(response)
            except urllib.error.HTTPError as error:
                return error.code, json.load(error)

        def check(name, passed):
            checks.append({'name': name, 'passed': bool(passed)})
            assert passed, name

        def start(restart=False):
            command = [sys.executable, __file__, '--server', '--database', str(database), '--port', str(port)]
            if restart:
                command.append('--restart')
            process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            processes.append(process)
            for _ in range(50):
                if process.poll() is not None:
                    raise RuntimeError('Synthetic server failed: ' + process.stderr.read().decode()[-2000:])
                try:
                    if call('/health')[0] == 200:
                        return process
                except (OSError, urllib.error.URLError):
                    time.sleep(.1)
            raise RuntimeError('Synthetic server did not become healthy')

        def stop(process):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)

        def add(rid, text, user='u1', session='s1'):
            return {'request_id': rid, 'user_id': user, 'session_id': session,
                    'messages': [{'role': 'user', 'content': text}]}

        def query(text='doctor', user='u1', top_k=100):
            return {'query': text, 'user_id': user, 'top_k': top_k}

        try:
            first = start()
            check('version_health', call('/health')[1]['version'] == 'text-http-hybrid-v8')
            check('missing_auth', call('/search', query(), key=None)[0] == 401)
            check('wrong_auth', call('/search', query(), key='wrong')[0] == 401)
            body = add('r1', 'I work as a physician.')
            status, accepted = call('/add', body)
            check('add_exact_echo', status == 200 and accepted == {'success': True, 'request_id': 'r1', 'user_id': 'u1', 'session_id': 's1'})
            check('idempotent_retry', call('/add', body) == (status, accepted))
            check('conflicting_retry', call('/add', add('r1', 'I work in a bakery.'))[0] == 409)
            status, result = call('/search', query())
            check('synthetic_semantic_search', status == 200 and 'physician' in result['data'][0]['content'])
            check('incremental_add', call('/add', add('r2', 'I work in a bakery.', session='s2'))[0] == 200)
            status, bread = call('/search', query('bread', top_k=1))
            check('incremental_search', status == 200 and 'bakery' in bread['data'][0]['content'])
            check('unknown_user_empty', call('/search', query(user='unknown')) == (200, {'data': []}))
            check('isolated_second_user', call('/add', add('r3', 'I work as a physician.', user='u2'))[0] == 200)
            check('failed_add_not_acknowledged', call('/add', add('failed', 'uncached synthetic body'))[0] == 503)
            check('failed_query_retryable', call('/search', query('uncached synthetic query'))[0] == 503)
            check('invalid_topk', call('/search', query(top_k=0))[0] == 422)
            _, saved = call('/search', query())
            stop(first)
            second = start(restart=True)
            check('restart_response_identical', call('/search', query()) == (200, saved))
            check('restart_idempotent_without_provider', call('/add', body) == (200, accepted))
            check('restart_query_vectors_reused', call('/search', query('bread', top_k=1)) == (200, bread))
            stop(second)
            with closing(sqlite3.connect(database)) as db:
                check('failed_write_left_no_raw_message_or_request',
                      db.execute('SELECT count(*) FROM messages').fetchone()[0] == 3 and
                      db.execute('SELECT count(*) FROM requests').fetchone()[0] == 3)
        finally:
            for process in processes:
                stop(process)
                process.stderr.close()
    print(json.dumps({'status': 'passed', 'checks': checks, 'server_processes': 2,
                      'model_api_calls': 0, 'vectors': 'synthetic fixtures only',
                      'network': 'loopback HTTP with synthetic vectors', 'official_platform_smoke': False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', action='store_true')
    parser.add_argument('--restart', action='store_true')
    parser.add_argument('--database', type=Path)
    parser.add_argument('--port', type=int)
    args = parser.parse_args()
    server(args.database, args.port, args.restart) if args.server else run()
