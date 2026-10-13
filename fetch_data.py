"""Cache public GBIF occurrence records and the original AVONET trait source."""
from pathlib import Path
import concurrent.futures
import datetime
import json
import urllib.parse
import urllib.request
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'data' / 'animals'
SPECIES = {'Cyanistes caeruleus': 2487879, 'Parus major': 9705453, 'Periparus ater': 2487871}

def get(url):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return r.read()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)

def fetch_species(item):
    species, key = item
    target = OUT / (species.replace(' ', '_') + '.json')
    if target.exists():
        print(species, 'cached', flush=True)
        return
    params = dict(taxonKey=key, hasCoordinate='true', hasGeospatialIssue='false',
                  occurrenceStatus='PRESENT', decimalLatitude='52.8998,53.2005',
                  decimalLongitude='-4.2002,-3.6997', year='2000,2025', limit=300)
    records = []
    for offset in range(0, 100000, 300):
        url = 'https://api.gbif.org/v1/occurrence/search?' + urllib.parse.urlencode({**params, 'offset':offset})
        data = json.loads(get(url))
        records.extend(data['results'])
        if offset == 0:
            print(species, 'GBIF count:', data['count'], flush=True)
        if data['endOfRecords']:
            break
    target.write_text(json.dumps(dict(species=species, query=params,
                      retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      count=len(records), records=records)), encoding='utf-8')
    print(species, len(records), 'saved', flush=True)

def fetch_traits():
    target = OUT / 'AVONET_Supplementary_dataset_1.xlsx'
    meta = json.loads(get('https://api.figshare.com/v2/articles/16586228'))
    (OUT / 'avonet_metadata.json').write_text(json.dumps(meta), encoding='utf-8')
    item = next(x for x in meta['files'] if x['name']=='AVONET Supplementary dataset 1.xlsx')
    if not target.exists():
        target.write_bytes(get(item['download_url']))
    print('AVONET saved', target.stat().st_size, flush=True)
    import pandas as pd
    frame = pd.read_excel(target, sheet_name='AVONET1_BirdLife')
    frame = frame[frame.Species1.isin(SPECIES)]
    if len(frame) != 3:
        raise ValueError('Could not match all three species in AVONET')
    frame.to_json(OUT/'avonet_three_tits.json', orient='records', indent=2)

if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        fs=[ex.submit(fetch_species, x) for x in SPECIES.items()]+[ex.submit(fetch_traits)]
        for f in concurrent.futures.as_completed(fs):
            f.result()
