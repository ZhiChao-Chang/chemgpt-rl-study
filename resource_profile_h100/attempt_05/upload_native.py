"""Upload one hashed input bundle through the job's native terminal, no WAN."""
from pathlib import Path
import base64
import fcntl
import json
import os
import pty
import re
import select
import shlex
import subprocess
import sys
import termios
import time
import tty
import zlib

ROOT=Path(__file__).resolve().parent
CONTROL=Path('/home/zhangzhichao/codex-cci-control')
sys.path.insert(0,str(CONTROL))
import acp_ctl as acp
_,j=acp.load(CONTROL/'state/acp/jobs/acp-explore-chemgpt-profile-20261104-05.json')
job=acp.describe(j)
if job['state']!='RUNNING':raise RuntimeError('Wait for this fixed job to be RUNNING')
workers=acp.workers_api(j)['workers']
running=[w for w in workers if w['phase']=='Running']
if len(running)!=1:raise RuntimeError('Exactly one running owned Worker is required')
w=running[0]
argv=[acp.SCO,'--profile',j['config']['sco_profile'],'--region',j['config']['region'],
      'acp','jobs','exec',j['task_id'],'--workspace-name',j['config']['workspace'],'--worker-name',w['name']]
master,slave=pty.openpty();tty.setraw(slave)
child=subprocess.Popen(argv,stdin=slave,stdout=slave,stderr=slave,close_fds=True,start_new_session=True)
os.close(slave);os.set_blocking(master,False)
tail=b'';start=time.monotonic()
def pump(timeout=.1):
    global tail
    if select.select([master],[],[],timeout)[0]:
        try:b=os.read(master,65536)
        except (BlockingIOError,OSError):return
        tail=(tail+b)[-1048576:]
def send(data):
    pos=0
    while pos<len(data):
        if time.monotonic()-start>480:raise TimeoutError('native transfer deadline')
        pump(0)
        if select.select([],[master],[],.2)[1]:
            try:pos+=os.write(master,data[pos:pos+16384])
            except BlockingIOError:pass
        if child.poll() is not None:raise RuntimeError('native client exited before acknowledgment')
def expect(marker,seconds):
    until=time.monotonic()+seconds
    while marker not in tail:
        pump(.2)
        if child.poll() is not None:raise RuntimeError('native client exited before marker')
        if time.monotonic()>until:raise TimeoutError('native marker deadline: '+marker.decode())
try:
    # Give the authenticated terminal time to attach before disabling shell echo.
    until=time.monotonic()+12
    while time.monotonic()<until:
        pump(.2)
        if re.search(rb'[#\$] ?',tail):break
        if child.poll() is not None:raise RuntimeError('terminal attach failed')
    send(b"stty -echo\nprintf 'NATIVE_SHELL_READY\\n'\n")
    expect(b'NATIVE_SHELL_READY',15)
    # Clear attach/echo output; only known transfer markers are exposed.
    tail=b''
    encoded=base64.b64encode(zlib.compress((ROOT/'native_receiver.py').read_bytes(),9)).decode()
    code="import base64,zlib;exec(zlib.decompress(base64.b64decode("+repr(encoded)+")))"
    command='/root/anaconda3/envs/ddc_encoding/bin/python -u -c '+shlex.quote(code)+'\n'
    if len(command)>3900:raise RuntimeError('receiver command exceeds safe terminal line length')
    send(command.encode());expect(b'NATIVE_BUNDLE_RECEIVER_READY',30)
    print(json.dumps({'phase':'receiver_ready','worker_uid':w['uid']}),flush=True)
    data=Path(json.loads((ROOT/'acp_profile_recipe.json').read_text())['bundle']['path']).read_bytes()
    sent=0
    for off in range(0,len(data),1536*128):
        block=data[off:off+1536*128]
        payload=b''.join(base64.b64encode(block[i:i+1536])+b'\n' for i in range(0,len(block),1536))
        send(payload);sent+=len(block)
        if sent%(1536*128*16)==0:print(json.dumps({'phase':'sending','compressed_bytes_sent':sent}),flush=True)
    send(b'__END_BUNDLE__\n');expect(b'NATIVE_BUNDLE_UPLOAD_COMPLETE',180)
    ack={'status':'UPLOADED_AND_REMOTE_HASH_VERIFIED','job_uid':job['uid'],'worker_uid':w['uid'],
         'compressed_bytes':len(data),'seconds':time.monotonic()-start,
         'transport':'SCO native terminal, shell echo disabled; base64 blocks under PTY line limit'}
    (ROOT/'native_upload_receipt.json').write_text(json.dumps(ack,indent=2));print(json.dumps(ack),flush=True)
    send(b'exit\n')
finally:
    if child.poll() is None:
        child.terminate()
        try:child.wait(timeout=5)
        except subprocess.TimeoutExpired:child.kill();child.wait()
    os.close(master)
