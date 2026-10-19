"""Regressao: uma unica geracao e contagem validada; sem GPU."""
import ast
import math
from pathlib import Path
import types
import unittest

BASE = Path(__file__).resolve().parents[1]

class Tensor:
    def __init__(self, data): self.data = data
    def tolist(self): return self.data
    def __getitem__(self, key): return Tensor(self.data[key])

class TimingTests(unittest.TestCase):
    def env(self, path, emitted=(10, 11, 99), returned=None):
        tree=ast.parse(path.read_text(encoding='utf-8'))
        nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))
               and n.name in {'TokenTimer','timed_generate','timed_answer'}]
        ticks=iter([0., .5, 1., 1.5, 1.6] if len(emitted)==3 else [0., .5, .6])
        calls=[]
        def generate(**kwargs):
            calls.append(kwargs)
            timer=kwargs['streamer']
            timer.put(Tensor([[1,2]]))
            for token in emitted: timer.put(Tensor([token]))
            return Tensor([[1,2]+list(emitted if returned is None else returned)])
        model=types.SimpleNamespace(generate=generate,generation_config=types.SimpleNamespace(eos_token_id=99))
        env=dict(torch=types.SimpleNamespace(inference_mode=lambda: lambda f:f),
                 BaseStreamer=object, time=types.SimpleNamespace(perf_counter=lambda:next(ticks)),
                 build_prompt=lambda q:(None,q),encode_prompt=lambda p,n:{'input_ids':Tensor([[1,2]])},
                 cuda_sync=lambda:None,postprocess_answer=lambda x:x,
                 text_tokenizer=types.SimpleNamespace(pad_token_id=0,eos_token_id=99,decode=lambda *a,**k:'answer'),
                 SPEED_MAX_NEW_TOKENS=160,model=model)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
        return env,calls

    def test_single_call_same_request_and_eos_at_cap(self):
        for p in (BASE/'modelos').glob('*.py'):
            env,calls=self.env(p)
            result=env['timed_generate'](env['model'],'q',3)
            self.assertEqual(len(calls),1)
            self.assertEqual(result['ttft_s'],.5)
            self.assertEqual(result['ttlt_s'],1.5)
            self.assertEqual(result['decode_tokens_per_s'],2.)
            self.assertEqual(result['new_tokens'],3)
            self.assertTrue(result['finished'])

    def test_one_token_no_decode_rate(self):
        for p in (BASE/'modelos').glob('*.py'):
            env,calls=self.env(p,emitted=(99,))
            result=env['timed_answer']('q',1)
            self.assertTrue(math.isnan(result['tokens_per_s']))
            self.assertEqual(result['ttft_s'],result['ttlt_s'])
            self.assertEqual(len(calls),1)
            self.assertEqual(calls[0]['min_new_tokens'],1)
            self.assertEqual(calls[0]['max_new_tokens'],1)

    def test_fixed_length_rejects_short_output(self):
        for p in (BASE/'modelos').glob('*.py'):
            env,_=self.env(p,emitted=(99,))
            with self.assertRaisesRegex(ValueError,'exatamente'):
                env['timed_answer']('q',160)

    def test_mismatch_rejected(self):
        for p in (BASE/'modelos').glob('*.py'):
            env,_=self.env(p,returned=(10,12,99))
            with self.assertRaisesRegex(ValueError,'Tokens do streamer'):
                env['timed_answer']('q',3)

    def test_prompt_or_chunk_mismatch_rejected(self):
        for p in (BASE/'modelos').glob('*.py'):
            env,_=self.env(p)
            timer=env['TokenTimer']([1,2])
            with self.assertRaises(ValueError): timer.put(Tensor([[3,4]]))
            timer=env['TokenTimer']([1,2]); timer.put(Tensor([[1,2]]))
            with self.assertRaises(ValueError): timer.put(Tensor([10,11]))

    def test_token_limit_without_eos(self):
        for p in (BASE/'modelos').glob('*.py'):
            env,_=self.env(p,emitted=(10,11,12))
            self.assertFalse(env['timed_generate'](env['model'],'q',3)['finished'])

if __name__=='__main__': unittest.main()
