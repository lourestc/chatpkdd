from pathlib import Path
import json

train_path = Path('TrainSplitted ')

channels = set()
users = set()
for fname in train_path.glob('*'):
  with open(fname) as f:
    for line in f:
      channels.add(json.loads(line)['c'])
      users.add(json.loads(line)['u'])

with open('channels','w') as f:
    for c in channels:
        f.write(c+'\n')

with open('users','w') as f:
    for u in users:
        f.write(u+'\n')
		
with open('channel-users','w') as fcu:
	for fname in train_path.glob('*'):
		with open(fname) as f:
			cu = {}
			for line in f:
				ckey = json.loads(line)['c']
				ukey = json.loads(line)['u']
				cu[ckey] = cu.get( ckey, [] )+[ukey]
		fcu.write(json.dumps(cu)+'\n')
			
cu = {}
with open('channel-users','r') as fcu:
	for line in fcu:
		for c,u in json.loads(line).items():
			cu[c] = cu.get(c,[])+u

with open('channel-users','w') as fcu:
	for c,u in cu.items():
		fcu.write(json.dumps({c:u})+'\n')

with open('user-channels','w') as fuc:
	for fname in train_path.glob('*'):
		with open(fname) as f:
			uc = {}
			for line in f:
				ckey = json.loads(line)['c']
				ukey = json.loads(line)['u']
				uc[ukey] = cu.get( ukey, [] )+[ckey]
		fuc.write(json.dumps(uc)+'\n')

uc = {}
with open('user-channels','r') as fuc:
	for line in fuc:
		for u,c in json.loads(line).items():
			uc[u] = uc.get(u,[])+c

with open('user-channels','w') as fuc:
	for u,c in uc.items():
		fuc.write(json.dumps({u:c})+'\n')