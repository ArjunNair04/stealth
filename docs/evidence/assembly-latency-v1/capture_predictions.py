import hashlib, importlib, json, struct, sys
from pathlib import Path
import numpy as np
import stealth_cpu as s
root=Path(__file__).resolve().parents[2]
out=Path(sys.argv[1]);out.mkdir(exist_ok=False)
blob=(root/'local/runtime-latest-v1/input.bin').read_bytes()
n,t,f=struct.unpack('<III',blob[8:20])
x=np.frombuffer(blob[20:],dtype='<f4').reshape(n,t,f)
results={}
for seed in (7,19,43):
 model=root/f'local/ettm1-correction-v1/seed{seed}/model.stlf'
 session=s.ForecastModel.from_bytes(model.read_bytes()).session()
 full=np.stack([session.predict(window) for window in x])
 prefixes=np.array([[session.predict_last(window[:i]) for i in range(1,t+1)] for window in x],dtype=np.float32)
 assert full.tobytes()==prefixes.tobytes(), 'Prefix causality changed'
 payload=full.astype('<f4').tobytes();(out/f'seed{seed}.bin').write_bytes(payload)
 results[str(seed)]={'sha256':hashlib.sha256(payload).hexdigest(),'values':int(full.size),
   'model_sha256':hashlib.sha256(model.read_bytes()).hexdigest()}
 print(seed,results[str(seed)],flush=True)
extension=Path(importlib.import_module('stealth_cpu._core').__file__)
report={'build_info':s.build_info(),'extension_sha256':hashlib.sha256(extension.read_bytes()).hexdigest(),
'input_sha256':hashlib.sha256(blob).hexdigest(),'results':results}
(out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
