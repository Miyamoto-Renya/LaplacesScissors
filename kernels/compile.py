"""Triton 컴파일 요청을 내용 주소의 장치 코드로 변환한다."""
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

sys.dont_write_bytecode = True
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
import ops


def build(spec):
    root = Path(spec['cache'])
    root.mkdir(parents=True, exist_ok=True)
    identity = dict(spec)
    identity.pop('cache')
    digest = hashlib.sha256()
    for data in (json.dumps(identity, sort_keys=True).encode(), Path(__file__).read_bytes(),
                 Path(ops.__file__).read_bytes(), triton.__version__.encode()):
        digest.update(data)
    stem = root / digest.hexdigest()
    suffix = '.cubin' if spec['backend'] == 'cuda' else '.hsaco'
    code = stem.with_suffix(suffix)
    receipt = stem.with_suffix('.json')
    if receipt.exists() and code.exists():
        return dict(json.loads(receipt.read_text()), binary=str(code))
    target = GPUTarget(spec['backend'], int(spec['architecture']) if spec['backend'] == 'cuda'
                       else spec['architecture'], spec['warp_size'])
    ast = ASTSource(getattr(ops, spec['operation']), spec['signature'], constexprs=spec['constants'])
    compiled = triton.compile(ast, target=target,
                             options={'num_warps': spec['num_warps'], 'enable_fp_fusion': False})
    metadata = compiled.metadata
    record = {'request': identity, 'name': metadata.name, 'shared': metadata.shared,
              'warps': metadata.num_warps}
    for key in ('global_scratch_size', 'global_scratch_align', 'profile_scratch_size', 'profile_scratch_align'):
        record[key] = getattr(metadata, key)
    temporary = Path(str(code) + '.' + str(os.getpid()) + '.partial')
    temporary.write_bytes(compiled.asm[suffix[1:]])
    temporary.replace(code)
    pending_receipt = Path(str(receipt) + '.' + str(os.getpid()) + '.partial')
    pending_receipt.write_text(json.dumps(record))
    pending_receipt.replace(receipt)
    return dict(record, binary=str(code))


for request in sys.stdin:
    try:
        response = build(json.loads(request))
    except Exception as exception:
        traceback.print_exc(file=sys.stderr)
        response = {'error': str(exception)}
    print(json.dumps(response), flush=True)
