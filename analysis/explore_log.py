from inspect_ai.log import read_eval_log

log = read_eval_log('logs/2026-03-02T04-03-21+00-00_manta-test5_aW96NoDj5Vrh6sHjpDXqTE.eval')
print('Status:', log.status)
print('Samples count:', len(log.samples) if log.samples else 0)

if log.samples:
    s = log.samples[0]
    print('Sample id:', s.id)
    print('Score keys:', list(s.scores.keys()) if s.scores else None)
    if s.scores:
        k = list(s.scores.keys())[0]
        sc = s.scores[k]
        print('Score value:', sc.value)
        print('Score metadata keys:', list(sc.metadata.keys()) if sc.metadata else None)
        if sc.metadata and 'dimensions' in sc.metadata:
            dims = sc.metadata['dimensions']
            print('Dimension keys:', list(dims.keys()))
            print('Sample dim entry:', dims[list(dims.keys())[0]])
