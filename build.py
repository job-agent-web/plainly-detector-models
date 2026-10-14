import json,hashlib,urllib.request,pathlib,shutil
root=pathlib.Path('site');root.mkdir(exist_ok=True)
models=json.load(open('models.json'))
for id,a in models.items():
 name=a['url'].split('/')[-1]
 urllib.request.urlretrieve('https://github.com/job-agent-web/plainly-detector-models/releases/download/browser-v1/'+name,root/name)
 assert (root/name).stat().st_size==a['bytes']
 assert hashlib.file_digest(open(root/name,'rb'),'sha256').hexdigest()==a['sha256']
for p in pathlib.Path('.').glob('*'):
 if p.suffix in ['.md','.txt','.json'] or p.name=='NOTICE':shutil.copy(p,root/p.name)
(root/'index.html').write_text('<h1>Open-source detector models</h1><p>Public model files and attribution only. No user text or inference service.</p><a href="README.md">Sources, licences and limitations</a>')
assert sum(p.stat().st_size for p in root.iterdir())<1000000000
