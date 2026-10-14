"""Create reproducible browser copies of public model weights. No remote code."""
from pathlib import Path
import json, logging, shutil, gc, hashlib, sys
import numpy as np
import torch
import onnx
from onnx import numpy_helper, helper, TensorProto
import onnxruntime as ort
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer, DefaultWeightOnlyQuantConfig
from onnxruntime.quantization import quantize_dynamic, QuantType
from transformers import AutoConfig, AutoModel, AutoTokenizer
from safetensors.torch import load_file

ROOT=Path(__file__).resolve().parent/'detector-models'
DEST=Path(__file__).resolve().parent/'detector-release'
DEST.mkdir(exist_ok=True)
logging.basicConfig(level=logging.WARNING)
logging.disable(logging.INFO)
torch.set_num_threads(4)
SAMPLES=[
 'We want to review the proposed workshop plans. The volunteer group will meet on Saturday to decide which supplies are needed.',
 'I left my notebook in the kitchen yesterday. Sam found it behind the kettle and brought it round after lunch, which saved me another trip.',
 'Effective collaboration plays a pivotal role in driving organizational success. By fostering a culture of innovation, teams can leverage diverse perspectives to achieve sustainable outcomes.',
]
def probability(kind,values):
 a=np.asarray(values).reshape(-1)
 if kind=='desklib':return float(1/(1+np.exp(-float(a[0]))))
 if kind=='tropa':
  e=np.exp(a-a.max());return float(1-e[0]/e.sum())
 return float(a[0])

def quantize_embeddings(model):
 """Gather int8 rows before decoding; never materialize a full fp32 table."""
 initializers={i.name:i for i in model.graph.initializer};new_nodes=[];converted=set()
 for node in model.graph.node:
  source=initializers.get(node.input[0]) if node.op_type=='Gather' else None
  if source is None or source.data_type!=TensorProto.FLOAT or len(source.dims)!=2 or source.dims[0]<10000:
   new_nodes.append(node);continue
  name=source.name
  if name not in converted:
   data=numpy_helper.to_array(source)
   scale=np.maximum(np.max(np.abs(data),axis=1)/127,np.finfo(np.float32).tiny).astype(np.float32)
   q=np.clip(np.rint(data/scale[:,None]),-127,127).astype(np.int8)
   model.graph.initializer.remove(source)
   model.graph.initializer.extend([numpy_helper.from_array(q,name+'.q8'),numpy_helper.from_array(scale,name+'.scale'),numpy_helper.from_array(np.array([-1],dtype=np.int64),name+'.axis')])
   converted.add(name)
  prefix=node.output[0]+'.decoded'
  opset=next(o.version for o in model.opset_import if o.domain=='')
  expand=helper.make_node('Unsqueeze',[prefix+'.s',name+'.axis'],[prefix+'.expanded']) if opset>=13 else helper.make_node('Unsqueeze',[prefix+'.s'],[prefix+'.expanded'],axes=[-1])
  new_nodes.extend([
   helper.make_node('Gather',[name+'.q8',node.input[1]],[prefix+'.q'],axis=0),
   helper.make_node('Gather',[name+'.scale',node.input[1]],[prefix+'.s'],axis=0),
   helper.make_node('Cast',[prefix+'.q'],[prefix+'.f'],to=TensorProto.FLOAT),
   expand,
   helper.make_node('Mul',[prefix+'.f',prefix+'.expanded'],list(node.output)),
  ])
 del model.graph.node[:];model.graph.node.extend(new_nodes)
 return len(converted)

def make_q4(kind,source):
 print(json.dumps({'id':kind,'stage':'quantizing'}),flush=True)
 model=onnx.load(source)
 embedding_count=quantize_embeddings(model)
 q=MatMulNBitsQuantizer(model,algo_config=DefaultWeightOnlyQuantConfig(block_size=128,is_symmetric=True,accuracy_level=4,bits=4,op_types_to_quantize=('MatMul',)))
 q.process()
 model=q.model.model
 model.doc_string='Browser conversion for Plainly; original model licences apply.'
 for node in model.graph.node:node.doc_string=''
 path=DEST/(kind+'-q4.onnx')
 onnx.save_model(model,str(path),save_as_external_data=False)
 del q,model;gc.collect()
 print(json.dumps({'id':kind,'stage':'converted','bytes':path.stat().st_size,'embedding_tables':embedding_count}),flush=True)
 return path

def make_q8(kind,source):
 print(json.dumps({'id':kind,'stage':'higher-precision-conversion'}),flush=True)
 model=onnx.load(source)
 constants={i.name for i in model.graph.initializer};aliases={};kept=[]
 for node in model.graph.node:
  if node.op_type=='Identity' and (node.input[0] in constants or node.input[0] in aliases):
   aliases[node.output[0]]=aliases.get(node.input[0],node.input[0])
  else:kept.append(node)
 for node in kept:
  for i,name in enumerate(node.input):node.input[i]=aliases.get(name,name)
 del model.graph.node[:];model.graph.node.extend(kept)
 if kind!='smollm':quantize_embeddings(model)
 intermediate=ROOT/kind/'embedding-q8.onnx'
 onnx.save_model(model,str(intermediate));del model;gc.collect()
 target=DEST/(kind+'-browser.onnx')
 quantize_dynamic(str(intermediate),str(target),weight_type=QuantType.QInt8,per_channel=True,op_types_to_quantize=['MatMul'],extra_options={'MatMulConstBOnly':True})
 print(json.dumps({'id':kind,'stage':'converted-int8','bytes':target.stat().st_size}),flush=True)
 return target

class Detector(torch.nn.Module):
 def __init__(self,config):
  super().__init__();self.model=AutoModel.from_config(config);self.classifier=torch.nn.Linear(config.hidden_size,getattr(config,'detector_num_labels',1))
 def forward(self,input_ids,attention_mask):
  hidden=self.model(input_ids=input_ids,attention_mask=attention_mask)[0]
  mask=attention_mask.unsqueeze(-1).float()
  return self.classifier((hidden*mask).sum(1)/mask.sum(1).clamp_min(1))

reports=[]
baseline_cache={r['id']:r['original'] for r in json.loads((DEST/'conversion-validation.json').read_text())} if (DEST/'conversion-validation.json').exists() else {}
for kind in (sys.argv[1:] or ['smollm','desklib','tropa']):
 root=ROOT/kind
 tokenizer=AutoTokenizer.from_pretrained(root,local_files_only=True,trust_remote_code=False)
 if kind=='smollm' and tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
 source=root/('fixed_optimized_detector.onnx' if kind=='smollm' else 'export.onnx')
 base_scores=[]
 if source.exists() and kind in baseline_cache:
  base_scores=baseline_cache[kind]
 elif kind!='smollm':
  print(json.dumps({'id':kind,'stage':'loading-original'}),flush=True)
  config=AutoConfig.from_pretrained(root,local_files_only=True,trust_remote_code=False)
  model=Detector(config)
  model.load_state_dict(load_file(root/'model.safetensors'),strict=True,assign=True)
  model.eval()
  with torch.inference_mode():
   for sample in SAMPLES:
    enc=tokenizer(sample,return_tensors='pt',truncation=True,max_length=768)
    base_scores.append(probability(kind,model(enc['input_ids'],enc['attention_mask']).numpy()))
  if not source.exists():
   print(json.dumps({'id':kind,'stage':'exporting'}),flush=True)
   enc=tokenizer(SAMPLES[0],return_tensors='pt')
   torch.onnx.export(model,(enc['input_ids'],enc['attention_mask']),str(source),input_names=['input_ids','attention_mask'],output_names=['logits'],dynamic_axes={'input_ids':{0:'batch',1:'seq'},'attention_mask':{0:'batch',1:'seq'},'logits':{0:'batch'}},opset_version=17,dynamo=False,do_constant_folding=True)
  del model;gc.collect()
 else:
  session=ort.InferenceSession(str(source),providers=['CPUExecutionProvider'])
  for sample in SAMPLES:
   enc=tokenizer(sample,return_tensors='np',truncation=True,padding='max_length',max_length=256)
   base_scores.append(probability(kind,session.run(None,{k:enc[k].astype(np.int64) for k in ['input_ids','attention_mask']})[0]))
  del session;gc.collect()
 target=DEST/(kind+'-browser.onnx')
 model=onnx.load(source)
 constants={i.name for i in model.graph.initializer};aliases={};kept=[]
 for node in model.graph.node:
  if node.op_type=='Identity' and (node.input[0] in constants or node.input[0] in aliases):aliases[node.output[0]]=aliases.get(node.input[0],node.input[0])
  else:kept.append(node)
 for node in kept:
  for i,name in enumerate(node.input):node.input[i]=aliases.get(name,name)
 del model.graph.node[:];model.graph.node.extend(kept)
 if kind!='smollm':quantize_embeddings(model)
 q=MatMulNBitsQuantizer(model,algo_config=DefaultWeightOnlyQuantConfig(block_size=32,is_symmetric=True,accuracy_level=4,bits=8,op_types_to_quantize=('MatMul',)))
 q.process();onnx.save_model(q.model.model,str(target));del q,model;gc.collect()
 print(json.dumps({'id':kind,'stage':'validating'}),flush=True)
 session=ort.InferenceSession(str(target),providers=['CPUExecutionProvider'])
 quant_scores=[]
 for sample in SAMPLES:
  options={'truncation':True,'max_length':768} if kind!='smollm' else {'truncation':True,'padding':'max_length','max_length':256}
  enc=tokenizer(sample,return_tensors='np',**options)
  quant_scores.append(probability(kind,session.run(None,{k:enc[k].astype(np.int64) for k in ['input_ids','attention_mask']})[0]))
 report={'id':kind,'original':base_scores,'browser':quant_scores,'max_probability_difference':max(abs(a-b) for a,b in zip(base_scores,quant_scores)),'bytes':target.stat().st_size,'sha256':hashlib.file_digest(open(target,'rb'),'sha256').hexdigest()}
 print(json.dumps(report),flush=True)
 if report['max_probability_difference']>0.05:raise RuntimeError('Browser conversion changed probabilities by more than 5 percentage points: '+kind)
 reports.append(report)
 (DEST/'conversion-validation.json').write_text(json.dumps(reports,indent=2))
 del session;gc.collect()

# SLOP q4 uses half-precision non-quantized operations. Convert these to float
# for the CPU/WASM backend; learned weights and feature normalization stay intact.
model=onnx.load(ROOT/'slop'/'model_q4.onnx')
def float_tensor(tensor):
 if tensor.data_type==TensorProto.FLOAT16:
  tensor.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(tensor).astype(np.float32),tensor.name))
for tensor in model.graph.initializer:float_tensor(tensor)
for value in [*model.graph.input,*model.graph.output,*model.graph.value_info]:
 if value.type.tensor_type.elem_type==TensorProto.FLOAT16:value.type.tensor_type.elem_type=TensorProto.FLOAT
for node in model.graph.node:
 for attr in node.attribute:
  if attr.type==onnx.AttributeProto.TENSOR:float_tensor(attr.t)
  if node.op_type=='Cast' and attr.name=='to' and attr.i==TensorProto.FLOAT16:attr.i=TensorProto.FLOAT
 node.doc_string=''
model.doc_string='SLOP browser CPU copy: FP16 operations converted to FP32. MIT.'
target=DEST/'slop-q4.onnx';onnx.save_model(model,str(target))
print(json.dumps({'id':'slop','stage':'converted','bytes':target.stat().st_size,'sha256':hashlib.file_digest(open(target,'rb'),'sha256').hexdigest()}),flush=True)
