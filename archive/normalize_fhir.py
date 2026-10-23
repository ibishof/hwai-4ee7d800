"""Deterministic, first-pass FHIR-to-compact-context normalizer.

Usage:
  python normalize_fhir.py data/synthea_sample_data_fhir_latest/Small.json
  python normalize_fhir.py path/to/Small.json --count-tokens

Requires Python 3.10+. Optional for exact token count: pip install anthropic
No LLM is used for summarization. Raw input should be retained separately.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any


def concept(v: Any) -> str:
    if not isinstance(v, dict):
        return str(v) if v is not None else ''
    if v.get('text'):
        return str(v['text'])
    codings = v.get('coding') or []
    if codings:
        return str(codings[0].get('display') or codings[0].get('code') or '')
    return ''


def quantity(v: Any) -> str:
    if not isinstance(v, dict):
        return ''
    val = v.get('value')
    if val is None:
        return ''
    return f"{val} {v.get('unit') or v.get('code') or ''}".strip()


def ref(v: Any) -> str:
    return v.get('reference', '') if isinstance(v, dict) else ''


def date_of(r: dict) -> str:
    for key in ('effectiveDateTime', 'occurrenceDateTime', 'performedDateTime', 'authoredOn', 'recordedDate', 'onsetDateTime', 'issued', 'date', 'created'):
        if r.get(key):
            return str(r[key])[:10]
    for key in ('period', 'effectivePeriod', 'performedPeriod', 'onsetPeriod'):
        if isinstance(r.get(key), dict) and r[key].get('start'):
            return str(r[key]['start'])[:10]
    return ''


def patient_line(r: dict) -> str:
    name = (r.get('name') or [{}])[0]
    full = ' '.join([*(name.get('given') or []), name.get('family', '')]).strip()
    return 'Patient: ' + '; '.join(filter(None, [full, f"sex/gender: {r.get('gender', '?')}", f"DOB: {r.get('birthDate', '?')}"]))


def observation_text(r: dict) -> str:
    label = concept(r.get('code')) or 'unnamed observation'
    value = quantity(r.get('valueQuantity'))
    if not value:
        for key in ('valueCodeableConcept', 'valueString', 'valueBoolean', 'valueInteger', 'valueDateTime'):
            if key in r:
                value = concept(r[key]) if key == 'valueCodeableConcept' else str(r[key])
                break
    components = []
    for c in r.get('component') or []:
        v = quantity(c.get('valueQuantity')) or concept(c.get('valueCodeableConcept')) or str(c.get('valueString') or '')
        components.append(f"{concept(c.get('code'))}: {v}")
    if components:
        value = '; '.join(components)
    flag = ', '.join(filter(None, (concept(x) for x in r.get('interpretation') or [])))
    range_text = []
    for x in r.get('referenceRange') or []:
        low, high = quantity(x.get('low')), quantity(x.get('high'))
        if low or high:
            range_text.append(f"{low}..{high}")
    tail = f" [{flag}]" if flag else ''
    tail += f" (ref: {', '.join(range_text)})" if range_text else ''
    return f"{label}: {value or '[no scalar value]'}{tail}"


def describe(r: dict) -> str:
    t = r.get('resourceType', 'Unknown')
    if t == 'Patient':
        return patient_line(r)
    if t == 'Encounter':
        kind = ', '.join(filter(None, (concept(x) for x in r.get('type') or []))) or (r.get('class') or {}).get('display', '')
        return f"Visit: {kind or 'unspecified'}; status={r.get('status','?')}; {r.get('period',{}).get('start','')[:10]}"
    if t == 'Observation':
        return observation_text(r)
    if t == 'Condition':
        status = concept(r.get('clinicalStatus'))
        verify = concept(r.get('verificationStatus'))
        return f"Condition: {concept(r.get('code'))}; status={status or '?'}; verification={verify or '?'}; onset={date_of(r) or '?'}"
    if t == 'Immunization':
        return f"Vaccine: {concept(r.get('vaccineCode'))}; status={r.get('status','?')}"
    if t == 'Procedure':
        return f"Procedure: {concept(r.get('code'))}; status={r.get('status','?')}"
    if t == 'DiagnosticReport':
        return f"Report: {concept(r.get('code'))}; status={r.get('status','?')}; results={len(r.get('result') or [])}"
    if t in ('MedicationRequest', 'MedicationStatement', 'MedicationAdministration', 'MedicationDispense'):
        med = concept(r.get('medicationCodeableConcept')) or r.get('medicationReference', {}).get('display', '')
        if not med and isinstance(r.get('medication'),dict):
            med = concept(r['medication']) or r['medication'].get('display','')
        doses=[]
        for instruction in r.get('dosageInstruction') or []:
            doses.append(instruction.get('text') or quantity((instruction.get('doseAndRate') or [{}])[0].get('doseQuantity')))
        return f"{t}: {med or '?'}; status={r.get('status','?')}" + (f"; dose={' | '.join(filter(None,doses))}" if doses else '')
    if t == 'AllergyIntolerance':
        return f"Allergy: {concept(r.get('code'))}; status={concept(r.get('clinicalStatus')) or '?'}; verification={concept(r.get('verificationStatus')) or '?'}"
    if t == 'DocumentReference':
        return f"Document: {concept(r.get('type')) or r.get('description','unspecified')}; status={r.get('status','?')} (content retained in source)"
    if t == 'Composition':
        return f"Composition: {concept(r.get('type'))}; title={r.get('title','')} (sections retained in source)"
    return f"{t}: [details retained in source]"


def load_resources(raw: Any) -> list[tuple[str, dict]]:
    if isinstance(raw, dict) and raw.get('resourceType') == 'Bundle':
        out=[]
        for entry in raw.get('entry') or []:
            r=entry.get('resource')
            if isinstance(r,dict):
                out.append((entry.get('fullUrl') or '',r))
        return out
    if isinstance(raw, list):
        return [('',r) for r in raw if isinstance(r,dict)]
    if isinstance(raw,dict):
        return [('',raw)]
    raise ValueError('Input must be FHIR Bundle, a single resource, or a resource list')


def normalize(path: Path, output: Path, count_tokens: bool) -> None:
    with path.open('r', encoding='utf-8-sig') as f:
        raw=json.load(f)
    entries=load_resources(raw)
    ids={}
    for i,(url,r) in enumerate(entries,1):
        key=f"R{i}"
        for candidate in (url, f"{r.get('resourceType','')}/{r.get('id','')}", r.get('id','')):
            if candidate:
                ids[candidate]=key
    records=[]
    by_encounter=defaultdict(list)
    for i,(url,r) in enumerate(entries,1):
        rid=f'R{i}'
        typ=r.get('resourceType','Unknown')
        encounter=ref(r.get('encounter'))
        encounter_id=ids.get(encounter, encounter)
        links={}
        for field in ('subject','patient','encounter','basedOn','partOf'):
            value=r.get(field)
            targets=value if isinstance(value,list) else [value]
            found=[ids.get(ref(v),ref(v)) for v in targets if ref(v)]
            if found: links[field]=found
        if typ=='DiagnosticReport':
            links['result']=[ids.get(ref(x),ref(x)) for x in r.get('result') or []]
        item={'id':rid,'type':typ,'date':date_of(r),'encounter':encounter_id,'text':describe(r),'links':links,
              'source_fullUrl':url,'source_id':r.get('id','')}
        records.append(item)
        if encounter_id and typ!='Encounter':
            by_encounter[encounter_id].append(item)
    # Explicitly label coverage to prevent omitted information being mistaken for absent findings.
    counts=Counter(x['type'] for x in records)
    lines=['# Patient clinical context (deterministic, preliminary)',
           'CAUTION: This is an index/overview, not a validated complete clinical summary.',
           'Unrepresented details remain in the original FHIR source; missing sections do not imply negative findings.',
           '', f"Resources: {len(records)}; types: "+', '.join(f'{k}={v}' for k,v in sorted(counts.items())), '']
    for x in records:
        if x['type']=='Patient': lines.append(f"[{x['id']}] {x['text']}")
    lines.extend(['','## Encounters (date order)'])
    visits=sorted((x for x in records if x['type']=='Encounter'),key=lambda x:(x['date'],x['id']))
    for visit in visits:
        lines.append(f"\n### [{visit['id']}] {visit['text']}")
        for x in sorted(by_encounter.get(visit['id'],[]), key=lambda x:(x['date'],x['id'])):
            if x['type'] in ('Claim','ExplanationOfBenefit','Provenance'):
                continue
            lines.append(f"- [{x['id']}] {x['text']}")
    lines.extend(['','## Not assigned to an encounter'])
    assigned={x['id'] for group in by_encounter.values() for x in group}
    for x in records:
        if x['type'] not in ('Patient','Encounter','Claim','ExplanationOfBenefit','Provenance') and x['id'] not in assigned:
            lines.append(f"- [{x['id']}] {x['date'] or '?'} {x['text']}")
    context='\n'.join(lines)+'\n'
    output.mkdir(parents=True,exist_ok=True)
    (output/'context.txt').write_text(context,encoding='utf-8')
    (output/'normalized.json').write_text(json.dumps({'records':records,'resource_counts':counts},indent=2,ensure_ascii=False),encoding='utf-8')
    stats={'input_file':str(path),'input_bytes':path.stat().st_size,'resource_count':len(records),
           'resource_counts':dict(counts),'context_bytes':len(context.encode('utf-8')),
           'context_characters':len(context),'context_words':len(context.split()),
           'character_reduction_pct':round(100*(1-len(context.encode('utf-8'))/max(1,path.stat().st_size)),2)}
    if count_tokens:
        try:
            from anthropic import Anthropic
            stats['opus5_tokens']=Anthropic().messages.count_tokens(
                model='claude-opus-5',messages=[{'role':'user','content':context}]).input_tokens
            stats['within_900k_limit']=stats['opus5_tokens']<=900000
        except Exception as exc:
            stats['token_count_error']=str(exc)
    (output/'stats.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    print(json.dumps(stats,indent=2))
    print('\nOutput:',output.resolve())
    print('\nPreview:\n', '\n'.join(lines[:35]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path)
    p.add_argument('--output',type=Path,default=Path('output')/'small_normalized')
    p.add_argument('--count-tokens',action='store_true')
    args=p.parse_args()
    normalize(args.input,args.output,args.count_tokens)

if __name__=='__main__':
    main()
