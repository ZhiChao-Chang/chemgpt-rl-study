#!/usr/bin/env python3
"""Download the recorded upstream model, without redistributing it in Git."""
import argparse,hashlib,json,pathlib,urllib.request
ROOT=pathlib.Path(__file__).resolve().parents[1]
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser(description='Download and SHA-256 check the recorded ChemGPT model; no training.')
 ap.add_argument('--destination',type=pathlib.Path,required=True);a=ap.parse_args()
 manifest=json.loads((ROOT/'proposal/evidence/download_manifest.json').read_text());a.destination.mkdir(parents=True,exist_ok=True)
 for spec in manifest['files']:
  assert pathlib.Path(spec['name']).name==spec['name']
  p=a.destination/spec['name']
  if p.is_file() and digest(p)==spec['sha256']:
   print(spec['name']+': verified existing file');continue
  tmp=p.with_name(p.name+'.part')
  try:
   with urllib.request.urlopen(spec['url'],timeout=120) as src,tmp.open('wb') as out:
    while chunk:=src.read(1024*1024):out.write(chunk)
   if tmp.stat().st_size!=spec['bytes'] or digest(tmp)!=spec['sha256']:raise ValueError('Download identity mismatch: '+spec['name'])
   tmp.replace(p);print(spec['name']+': downloaded and verified')
  finally:
   if tmp.exists():tmp.unlink()
 print('Model revision: '+manifest['revision'])
if __name__=='__main__':main()
