"""Conservative public-family aliases in addition to declared dataset groups.

Netlib describes GREENBEA/GREENBEB as the same refinery model. Its PILOT and
STAND variants also require grouping beyond stripping numeric suffixes.
Source: https://www.netlib.org/lp/data/readme (problem notes and sources).
This curated check is not a claim that every semantic relation is detectable.
"""
from collections import Counter
import re


def family(name):
    name=name.lower()
    if name in {'greenbea','greenbeb'}:return 'greenbe'
    if name.startswith('pilot'):return 'pilot'
    if name in {'standata','standgub','standmps'}:return 'stand'
    return re.split(r'[0-9]',name,maxsplit=1)[0].rstrip('-_.') or name


def audit_split(training, heldout, *, prior_names=()):
    def key(case):
        return 'netlib:'+family(case['name']) if case['source']=='netlib' else case['group']
    training_groups={key(c) for c in training}
    heldout_groups=[key(c) for c in heldout]
    prior_netlib={'netlib:'+family(n) for n in prior_names}
    overlaps=sorted((training_groups|prior_netlib)&set(heldout_groups))
    duplicates=sorted(k for k,n in Counter(heldout_groups).items() if n>1)
    hashes=sorted({c['data_hash'] for c in training}&{c['data_hash'] for c in heldout})
    return {'passed':not (overlaps or duplicates or hashes),'overlapping_families':overlaps,
            'duplicate_heldout_families':duplicates,'overlapping_data_hashes':hashes,
            'scope':'declared groups plus curated Netlib aliases; not exhaustive semantic independence'}
