"""Synthetic real TCP checks, two process lifetimes, no external network."""
import argparse
import base64
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from memory_bench.service.offline import loopback_only
from memory_bench.service.multitrack import VERSION

PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aH1sAAAAASUVORK5CYII="


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    key = secrets.token_urlsafe(32)
    env = {k:v for k,v in os.environ.items() if k not in {"AML_MEMORY_SYSTEM_KEY_FILE","OPENAI_API_KEY","DASHSCOPE_API_KEY","BAILIAN_API_KEY","HTTP_PROXY","HTTPS_PROXY","ALL_PROXY"}}
    env.update(AML_MEMORY_SYSTEM_KEY=key, AML_RETRIEVER="lexical", AML_ALLOW_EMBEDDING_API="0",
               AML_TEXT_DATABASE=str((output / "memory.sqlite3").resolve()), AML_CODE_IDENTIFIERS="1", OPENBLAS_NUM_THREADS="1")
    checks = []
    source = [{"type":"text","text":"The copper telescope is stored in the attic."},
              {"type":"image_url","image_url":{"url":PNG}}, {"type":"text","text":"Original note after the picture."}]
    with loopback_only() as network:
        for cycle in range(2):
            with socket.socket() as probe:
                probe.bind(("127.0.0.1",0))
                port = probe.getsockname()[1]
            base = f"http://127.0.0.1:{port}"
            def call(path, body=None, auth=True):
                request = urllib.request.Request(base+path, data=None if body is None else json.dumps(body).encode(),
                    headers={"Content-Type":"application/json",**({"Authorization":"Bearer "+key} if auth else {})})
                try:
                    with urllib.request.urlopen(request,timeout=20) as response:
                        return response.status,json.load(response)
                except urllib.error.HTTPError as error:
                    return error.code,json.loads(error.read())
            def check(name, condition):
                assert condition, name
                checks.append(name)
            log = (output / f"server-{cycle}.log").open("w")
            process = subprocess.Popen([sys.executable,"-m","memory_bench.service.multitrack","--port",str(port)],
                cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
            try:
                deadline=time.monotonic()+25
                while True:
                    try:
                        if call("/health",auth=False)[0]==200: break
                    except (OSError,ValueError): pass
                    if process.poll() is not None or time.monotonic()>deadline: raise RuntimeError("Server startup failed")
                    time.sleep(.1)
                check("health_"+str(cycle),call("/health",auth=False)[1]["version"]==VERSION)
                request={"request_id":"r1","user_id":"media","session_id":"s1","messages":[{"role":"user","content":source}]}
                query={"query":[{"type":"text","text":"copper telescope"}],"user_id":"media","top_k":100}
                if cycle==0:
                    check("auth",call("/add",request,False)[0]==401)
                    check("durable_add",call("/add",request)[0]==200)
                    first=call("/search",query)
                    check("ordered_original_evidence",first[0]==200 and first[1]["data"][0]["content"]==source)
                    check("idempotent_add",call("/add",request)[0]==200)
                    changed=json.loads(json.dumps(request)); changed["messages"][0]["content"][0]["text"]="altered"
                    check("conflict",call("/add",changed)[0]==409)
                    check("isolation",call("/search",{**query,"user_id":"other"})[1]=={"data":[]})
                    check("image_only_query",call("/search",{**query,"query":[source[1]]})[1]["data"][0]["content"]==source)
                    plain={"request_id":"r2","user_id":"text","session_id":"s2","messages":[{"role":"user","content":"The river flows east."}]}
                    check("text_add",call("/add",plain)[0]==200)
                    check("text_search",bool(call("/search",{"query":"river","user_id":"text","top_k":1})[1]["data"]))
                    code={"request_id":"r3","user_id":"code","session_id":"s3","messages":[{"role":"user","content":"diff --git a/router.py b/router.py\n@@ -1 +1 @@\n+def requestRouter(): return True"}]}
                    check("coding_add",call("/add",code)[0]==200)
                    check("coding_search",bool(call("/search",{"query":"request router","user_id":"code","top_k":1})[1]["data"]))
                    check("invalid_topk",call("/search",{**query,"top_k":101})[0]==422)
                    check("remote_image_rejected",call("/search",{**query,"query":[{"type":"image_url","image_url":{"url":"https://example.com/x.png"}}]})[0]==422)
                else:
                    check("restart_exact",call("/search",query)==first)
                    check("restart_idempotent",call("/add",request)[0]==200)
                    extra={**request,"request_id":"r4","messages":[{"role":"assistant","content":"The telescope has moved to the observatory."}]}
                    check("incremental_after_restart",call("/add",extra)[0]==200)
                    check("incremental_visible",any("observatory" in json.dumps(r["content"]) for r in call("/search",{**query,"query":"observatory"})[1]["data"]))
            finally:
                process.terminate()
                try: process.wait(timeout=15)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
                log.close()
        report={"status":"passed","checks":checks,"count":len(checks),"processes":2,"model_api_calls":0,"network":network}
        (output / "summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
        print(json.dumps(report),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    run(parser.parse_args().output)
