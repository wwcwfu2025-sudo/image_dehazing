"""Verify release bytes, the exact paper method scope, and displayed metrics."""
import csv
import hashlib
import json
import math
from pathlib import Path

STAMP = '2026100210'
EXPECTED_METHODS = {
    'M01':'DCP', 'M02':'CAP', 'S12':'DehazeNet', 'M07':'AOD-Net',
    'M10':'GridDehazeNet', 'M32':'WBPGNDN', 'M12':'FFA-Net',
    'M19':'MB-TaylorFormer', 'S01':'PromptIR', 'S14':'MixDehazeNet',
    'ODCR-D696':'ODCR-Mamba',
}
MODEL_HASH = 'D696BBFA42DD1A840708B9C42E824036862ED17D538B2B47551641D8145854DB'


def require(condition, message):
    if not condition:
        raise SystemExit('Verification failed: ' + message)


def main():
    root = Path(__file__).resolve().parent
    manifest_name = 'FILE_SHA256_' + STAMP + '.json'
    manifest = json.loads((root / manifest_name).read_text(encoding='utf-8'))
    expected = set(manifest['files']) | {manifest_name}
    # Verify the publication directory strictly, including any nested outputs.
    # Keep personal inputs, runtime outputs and environments outside this folder.
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.relative_to(root).parts[0] != '.git'}
    require(actual == expected, 'missing or unlisted files: ' + repr(sorted(actual ^ expected)))
    for relative, entry in manifest['files'].items():
        path = (root / relative).resolve()
        require(root in path.parents and not path.is_symlink(), 'unsafe release path ' + relative)
        raw = path.read_bytes()
        require(len(raw) == entry['bytes'] and hashlib.sha256(raw).hexdigest().upper() == entry['sha256'], 'file hash mismatch: ' + relative)

    def read_json(relative):
        return json.loads((root / relative).read_text(encoding='utf-8'))

    def records(relative):
        with (root / relative).open(encoding='utf-8-sig', newline='') as handle:
            return list(csv.DictReader(handle))

    def matrix(relative):
        with (root / relative).open(encoding='utf-8-sig', newline='') as handle:
            return list(csv.reader(handle))

    registry = read_json('results/PAPER_METHODS_' + STAMP + '.json')
    require({r['method_id']:r['paper_name'] for r in registry['methods']} == EXPECTED_METHODS and len(registry['methods']) == 11, 'method registry must contain exactly the paper 11 methods')
    full = read_json('results/main_tables/main_tables_fullprecision_' + STAMP + '.json')
    tables = full['tables']
    require(set(tables) == {'I','II','III','IV','V','VI','VII','VIII'}, 'unexpected main table set')
    expected_rows = {}
    fmt = lambda value: format(float(value), '.3f')
    for label in ('I','II'):
        expected_rows[label] = [[r['method']] + [fmt(r['case_values'][case]) for case in tables[label]['case_order']] + [fmt(r['mean_14'])] for r in tables[label]['rows']]
    expected_rows['III'] = [[r['method'],str(r['n'])] + [fmt(r[k]) for k in ['PSNR','SSIM','LPIPS','DISTS']] for r in tables['III']['rows']]
    iv = tables['IV']
    index = {(r['method_id'],r['dataset_id']):r for r in iv['rows']}
    expected_rows['IV'] = [[name] + ['/'.join(fmt(index[mid,ds][k]) for k in iv['packed_cell_metric_order']) for ds in iv['dataset_order']] for mid,name in EXPECTED_METHODS.items()]
    for label in ('I','II','III','IV'):
        require([r[0] for r in expected_rows[label]] == list(EXPECTED_METHODS.values()), 'comparison method ordering ' + label)
        for r in tables[label]['rows']:
            require(EXPECTED_METHODS.get(r['method_id']) == r['method'], 'name/ID mismatch in main table ' + label)
    for label, rows in expected_rows.items():
        require(matrix(tables[label]['display_file'])[1:] == rows, 'CSV/full-precision mismatch in Table ' + label)
    for label in ('V','VI','VII','VIII'):
        display = matrix(tables[label]['display_file'])[1:]
        require(len(display) == len(tables[label]['rows']), 'row count ' + label)
        for shown,r in zip(display,tables[label]['rows']):
            if label == 'V':
                vals = [fmt(r[k]) for k in ['PSNR','SSIM_Gaussian11','LPIPS','DISTS']]
                require(shown == [r['dataset_name'],str(r['n'])] + vals, 'Table V values/scene')
                require((r['method_id'],r['method_name']) == ('ODCR-D696','ODCR-Mamba'), 'Table V fixed method')
            elif label == 'VI':
                vals = [fmt(r[k] * (1000 if k in ('GMAE','RGBMAE') else 1)) for k in ['PSNR','SSIM','GMAE','RGBMAE']]
                require(shown == [r['difficulty'],str(r['n']),r['input']] + vals and r['input'] in ('Hazy','ODCR-Mamba'), 'Table VI values/control')
            elif label == 'VII':
                require(shown == [r['method']] + [fmt(r[k]) for k in ['keypoints','correct_3','precision_3','recovery_3']], 'Table VII values')
                require((r['method_id'],r['method']) in [('Input','Hazy'),('ODCR-D696','ODCR-Mamba')], 'Table VII control')
            else:
                require(shown == [r['branch_name'],str(r['n'])] + [fmt(r[k]) for k in ['psnr','ssim','lpips','dists']], 'Table VIII values')
                require(r['branch_id'] in ('direct','physical','output'), 'Table VIII branch')

    schema = read_json('results/RESULTS_SCHEMA_' + STAMP + '.json')
    observed = set()
    total_rows = 0
    for relative, info in schema['per_image_exports'].items():
        rows = records(relative)
        require(len(rows) == info['data_rows'], 'record count ' + relative)
        require({r['method'] for r in rows} == set(info['methods']), 'methods in ' + relative)
        keys = set()
        for r in rows:
            require(EXPECTED_METHODS.get(r['method_id']) == r['method'], 'unlisted method or name/ID mismatch: ' + relative)
            if r['method_id'] == 'ODCR-D696':
                require(r['checkpoint_sha256'] == MODEL_HASH, 'fixed model identity in ' + relative)
            key = (r['method_id'],r['dataset_id'],r['pair_key'],r.get('branch',''))
            require(key not in keys, 'duplicate record in ' + relative)
            keys.add(key)
            observed.add(r['method'])
            for k in ('PSNR','SSIM_Gaussian11','SSIM_uniform11','LPIPS','DISTS'):
                if k in r:
                    require(math.isfinite(float(r[k])), 'nonfinite metric in ' + relative)
        total_rows += len(rows)
    require(observed == set(EXPECTED_METHODS.values()), 'missing paper methods in per-image exports')
    expected_csv = {t['display_file'] for t in tables.values()} | set(schema['per_image_exports'])
    actual_csv = {p.relative_to(root).as_posix() for p in (root/'results').rglob('*.csv')}
    require(expected_csv == actual_csv, 'unregistered results CSV')
    expected_results = expected_csv | {'results/' + name + '_' + STAMP + '.json' for name in ('PAPER_METHODS','RESULTS_SCHEMA')} | {'results/main_tables/main_tables_fullprecision_' + STAMP + '.json'}
    require({p.relative_to(root).as_posix() for p in (root/'results').rglob('*') if p.is_file()} == expected_results, 'unregistered results file')

    def inspect_method_fields(value, context):
        if isinstance(value, dict):
            if 'method_id' in value:
                mid = value['method_id']
                require(mid in EXPECTED_METHODS or (mid == 'Input' and '/tables/VII/' in context + '/'), 'unlisted JSON method ID: ' + context)
                for field in ('method','method_name','paper_name'):
                    if field in value:
                        require(value[field] == ('Hazy' if mid == 'Input' else EXPECTED_METHODS[mid]), 'JSON name/ID mismatch: ' + context)
            for key,item in value.items():
                if key in ('method','method_name','paper_name') and isinstance(item,str):
                    require(item in EXPECTED_METHODS.values() or (item == 'Hazy' and '/tables/VII/' in context + '/'), 'unlisted JSON method name: ' + context)
                if key == 'methods' and isinstance(item,list) and all(isinstance(v,str) for v in item):
                    require(set(item) <= set(EXPECTED_METHODS.values()), 'unlisted JSON method collection: ' + context)
                inspect_method_fields(item, context + '/' + key)
        elif isinstance(value,list):
            for i,item in enumerate(value):
                inspect_method_fields(item, context + '/' + str(i))

    for path in (root/'results').rglob('*.json'):
        inspect_method_fields(read_json(path.relative_to(root)), path.relative_to(root).as_posix())
    prov = read_json('PROVENANCE_RESULTS.json')
    for relative, entry in prov['outputs'].items():
        raw = (root/relative).read_bytes()
        require(hashlib.sha256(raw).hexdigest().upper() == entry['sha256'] and len(raw) == entry['bytes'], 'provenance output mismatch ' + relative)
    print('PASS: {} file hashes; exactly 11 methods (10 external + ODCR-Mamba); 8 main tables; {} stored per-image rows (overlaps are documented).'.format(len(manifest['files']),total_rows))


if __name__ == '__main__':
    main()

# Generated (Beijing, to hour): 2026-10-02 10:00 +08:00
