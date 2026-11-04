import sys, numpy as np, torch
sys.path.insert(0,'scripts')
import multiseed_mlp as M
dev='cuda' if torch.cuda.is_available() else 'cpu'
df=M.load_base_df('data'); tr,va,te=M.split_real_data(df); train=M.apply_condition_to_train(tr,'augmented','data')
texts=train['comment_text'].tolist()[:384]
c=np.load('results/multiseed_cache/qwen_augmented_cached.npz')['X_tr'][:384]
a,_=M.embed_qwen(texts,dev,'assets'); b,_=M.embed_qwen(texts,dev,'assets')
print('device',dev,'cuda name',torch.cuda.get_device_name(0) if dev=='cuda' else '-')
print('run1 vs run2 (same process): max|diff| =',float(np.abs(a-b).max()),' identical =',bool(np.array_equal(a,b)))
print('run1 vs cached (earlier job): max|diff| =',float(np.abs(a-c).max()),' identical =',bool(np.array_equal(a,c)))
print('mean |diff| run1 vs cached =',float(np.abs(a-c).mean()),' typical |value| =',float(np.abs(c).mean()))
print('cosine min run1 vs cached =',float((np.sum(a*c,1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(c,axis=1))).min()))
