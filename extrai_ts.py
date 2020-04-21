from pathlib import Path
import json
import csv

subscribed = {}
with open('train_truth.csv') as f:
    csv_reader = csv.DictReader(f)
    for line in csv_reader:
        subscribed.setdefault( line['channel'], {} )[line['user']] = line['subscribed']

train_path = Path('TrainSplitted ')

for fname in train_path.glob('*'):
	with open('timestamps/'+fname.stem+'.csv','w') as f2:

		f2.write( 'channel,user,concatenated_m,delta_ts,subscribed\n' )
		
		with open(fname) as f:
			for line in f:

				uc = json.loads(line)
				
				ckey = uc['c']
				ukey = uc['u']
				messages = sorted( uc['ms'], key=lambda m:m['t'] )
	
				concatenatedm = ' '.join([ m['m'] for m in messages ])
				
				ts_list = [ m['t'] for m in messages ]
				delta_ts = [0]+[ ts2-ts1 for ts1,ts2 in zip(ts_list[:-1],ts_list[1:]) ]
				
				f2.write( ckey+','+ukey+','+concatenatedm+','+json.dumps(delta_ts)+','+subscribed[ckey][ukey]+'\n' )