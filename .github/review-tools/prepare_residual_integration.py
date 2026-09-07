"""One-off owner-requested staging builder; never merged into main.

Reads one fixed source-review artifact. Publishes only a new staging branch,
never updates main, existing PR heads, repository settings, or dataset payloads.
"""
from pathlib import Path
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile

REPO = 'IPS-Stuttgart/BayesianPhysTwin'
BASE = '9d7383ea56a0a9e3ad6753d1c42fe653cd7e615d'
BASE_TREE = 'eb230ad3508c84f635fe2fd03b5ff90e1d1a1248'
TARGET = 'maintenance/consolidate-residual-code-2026-09-07'
ARTIFACT = 10014138077
ARCHIVE_SHA = '9ab8a9d36c05b8b3cd6d4ed737aeb5ae837dda6235975e982543ab6f781c7194'
HERE = Path(__file__).resolve().parent
WORK = Path(os.environ['RUNNER_TEMP']) / 'residual-integration'
WORK.mkdir(exist_ok=False)
SOURCE = WORK / 'review'
SOURCE.mkdir()
STAGE = WORK / 'stage'
STAGE.mkdir()


def api(endpoint, payload=None):
    command = ['gh', 'api', f'repos/{REPO}/{endpoint}']
    if payload is not None:
        command += ['--method', 'POST', '--input', '-']
    return json.loads(subprocess.check_output(command, input=None if payload is None else json.dumps(payload).encode()))


def digest(data):
    return hashlib.sha256(data).hexdigest()


raw = subprocess.check_output(['gh', 'api', f'repos/{REPO}/actions/artifacts/{ARTIFACT}/zip'])
assert digest(raw) == ARCHIVE_SHA
archive = WORK / 'review.zip'
archive.write_bytes(raw)
with zipfile.ZipFile(archive) as z:
    for name in z.namelist():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
    z.extractall(SOURCE)
for name, sha in json.loads((SOURCE / 'sha256.json').read_text()).items():
    assert digest((SOURCE / name).read_bytes()) == sha, name
assert json.loads((SOURCE / 'base.json').read_text())['commit'] == BASE
with tarfile.open(SOURCE / 'main.tar.gz') as tar:
    tar.extractall(STAGE, filter='data')
assert len(list((STAGE / '.github/workflows').glob('*.yml'))) == 90
original = {str(p.relative_to(STAGE)): p.read_bytes() for p in STAGE.rglob('*') if p.is_file()}
info = {
    950: ('dirichlet-processes', 'blockwise-regimes', 'DLO4/DLO5 blockwise conditional residual regimes', None, 'experiments/dp_residual_regimes_dev_v1', ['test_model.py']),
    952: ('gaussian-processes', 'dlo2-nystrom', 'DLO2 matched-rank Nystrom GP', 202, '.', ['experiments/deform_gp_residual_dev_v1/test_core.py']),
    953: ('gaussian-processes', 'dlo1-additive', 'DLO1 additive independent/material GP', 194, '.', ['tests/test_deform_gp_residual_v1.py']),
    955: ('gaussian-processes', 'dlo2-data-efficiency', 'DLO2 subset-conditioned GP learning curves', 195, '.', ['tests/test_gaussian_residual_v1.py']),
    956: ('dirichlet-processes', 'dlo1-summary-gate', 'DLO1 trajectory-summary mixture and weighted experts', 203, '.', ['tests/test_dp_residual_pilot.py']),
    957: ('dirichlet-processes', 'query-surrogate', 'DP versus RBF/finite controls on the query surrogate', None, 'experiments/deform_dp_residual_v1', ['test_run.py']),
    958: ('dirichlet-processes', 'dlo45-source-experts', 'Source-only DLO4/DLO5 full-feature residual experts', None, '.', ['tests/test_dp_residual_source_screen_v1.py']),
    959: ('gaussian-processes', 'dlo2-sparse', 'DLO2 sparse variational GP comparison', 198, 'experiments/deform_gp_residual_pilot_v1', []),
    960: ('dirichlet-processes', 'dlo1-gated-experts', 'DLO1 joint-feature/residual gated experts', 197, '.', ['tests/test_dp_residual_development_v1.py']),
    961: ('dirichlet-processes', 'dlo1-sticky-hdp', 'DLO1 sticky-HDP AR(2) residual dynamics', 200, 'experiments/dp_residual_source_v1', ['test_model.py', 'test_adapter.py']),
    962: ('dirichlet-processes', 'dlo1-kinematic-gate', 'DLO1 kinematic mixture gate', 199, '.', ['experiments/dp_residual_kinematic_v1/test_model.py']),
    963: ('gaussian-processes', 'dlo2-material-gp', 'DLO2 material-coordinate GP', None, '.', ['tests/test_material_gp_development_v1.py']),
}
manifest = {'schema_version': 1, 'review_date': '2026-09-07', 'base_revision': BASE,
            'scope': 'research-only integration; no production-default or scientific-outcome change',
            'source_snapshot_run_id': 34110478996, 'source_snapshot_artifact_id': ARTIFACT,
            'experiments': [], 'deferred': [
                {'pr': 944, 'reason': 'earlier incomplete overlapping prototype without focused tests'},
                {'pr': 947, 'reason': 'earlier separate source-preparation route without focused tests'},
                {'pr': 964, 'reason': 'prospective temporal-discrepancy extension; native adapter not validated here'}]}
python_paths = []
retained_paths = set()
parents = [BASE]
for number, (family, identity, title, paper, cwd, tests) in info.items():
    metadata = json.loads((SOURCE / f'pr-{number}.json').read_text())
    sha = metadata['head_sha']
    parents.append(sha)
    pr_root = SOURCE / f'pr-{number}'
    pr_root.mkdir()
    with tarfile.open(SOURCE / f'pr-{number}.tar.gz') as tar:
        tar.extractall(pr_root, filter='data')
    study = {'pr': number, 'id': identity, 'family': family, 'title': title,
             'source_revision': sha, 'paper_pr': paper, 'files': [],
             'test_working_directory': cwd, 'test_paths': tests,
             'status': 'retained research implementation, not production promotion'}
    if number == 959:
        study['self_test_script'] = 'gp.py'
    for entry in metadata['files']:
        old = entry['filename']
        assert entry['status'] == 'added'
        data = (pr_root / old).read_bytes()
        assert hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == entry['sha']
        record = {'source_path': old, 'source_git_blob': entry['sha'], 'source_sha256': digest(data)}
        new = old
        if old.startswith(('.github/workflows/', '.github/requests/')):
            new = f'archive/research-workflows/residual-models-2026-09-07/pr-{number}/{old}'
            record['disposition'] = 'inactive-execution-history'
        elif old.endswith('.md') or '/results/' in old or old.startswith('results/') or old.endswith('result_summary.json'):
            record['disposition'] = 'historical-document-reference' if old.endswith('.md') else 'historical-outcome-reference'
            record['historical_url'] = f'https://github.com/{REPO}/blob/{sha}/{old}'
            study['files'].append(record)
            continue
        else:
            record['disposition'] = 'research-source'
            if number == 962:
                new = old.replace('dp_residual_development_v1', 'dp_residual_kinematic_v1')
                data = data.replace(b'experiments.dp_residual_development_v1', b'experiments.dp_residual_kinematic_v1')
            if new.endswith('.py'):
                python_paths.append(new)
        path = STAGE / new
        assert not path.exists() and new not in retained_paths, new
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        retained_paths.add(new)
        record['path'] = new
        study['files'].append(record)
    manifest['experiments'].append(study)
os.chdir(STAGE)
subprocess.run(['ruff', 'format', *python_paths], check=True)
subprocess.run(['ruff', 'check', '--select', 'I', '--fix', *python_paths], check=True)
for name in python_paths:
    path = STAGE / name
    text = path.read_text()
    tree = ast.parse(text)
    offsets = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    edits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'zip' and not any(k.arg == 'strict' for k in node.keywords):
            pos = offsets[node.end_lineno - 1] + node.end_col_offset - 1
            prefix = text[:pos].rstrip()
            edits.append((pos, ' strict=False' if prefix.endswith(',') else ', strict=False'))
    for pos, addition in sorted(edits, reverse=True):
        text = text[:pos] + addition + text[pos:]
    if name == 'experiments/deform_dp_residual_v1/reference.py':
        text = text.replace('def objective(setting):', 'def objective(setting, arm=arm):')
    if name == 'experiments/dp_residual_kinematic_v1/run.py':
        text = text.replace('from experiments.dp_residual_kinematic_v1.model import GateSpec, ResidualExperts', 'from experiments.dp_residual_kinematic_v1.model import (  # noqa: E402 -- source-tree bootstrap\n    GateSpec, ResidualExperts,\n)')
    if name == 'scripts/remote/run_dp_residual_cached_pilot.py':
        text = text.replace('import run_dp_residual_dlo1_pilot as m', 'import run_dp_residual_dlo1_pilot as m  # noqa: E402 -- source-tree bootstrap')
        text = text.replace('from bayesian_phystwin_experiments.deform_dlo_local_residual import (', 'from bayesian_phystwin_experiments.deform_dlo_local_residual import (  # noqa: E402 -- source-tree bootstrap')
    if name == 'tests/test_gaussian_residual_v1.py':
        text = text.replace('from run_gp_residual_development_v1 import', 'from scripts.remote.run_gp_residual_development_v1 import')
    path.write_text(text)
subprocess.run(['ruff', 'check', '--select', 'I', '--fix', *python_paths], check=True)
subprocess.run(['ruff', 'format', *python_paths], check=True)
subprocess.run(['ruff', 'check', *python_paths], check=True)
expected = json.loads((HERE / 'expected_code_hashes.json').read_text())
assert set(expected) == set(python_paths)
for name, sha in expected.items():
    assert digest((STAGE / name).read_bytes()) == sha, ('reviewed-code mismatch', name)
for study in manifest['experiments']:
    for item in study['files']:
        if 'path' in item:
            item['retained_sha256'] = digest((STAGE / item['path']).read_bytes())
            item['change'] = 'byte-identical' if item['retained_sha256'] == item['source_sha256'] else 'format/import/docstring hygiene; explicit original zip semantics; synchronous closure binding; namespace/test import repair only'
            if item['path'].endswith('.py'):
                item['numerical_ast_preserved'] = True
                item['review_basis'] = 'exact digest match to locally tested source with independently compared canonical numerical AST'
# Approved new text and two explicitly reviewed infrastructure modifications.
payload = json.loads((HERE / 'integration_payload.json').read_text())
allowed_existing = {'.github/workflows/changed-python-preflight.yml', 'tests/test_workflow_inventory_budget.py'}
for name, text in payload.items():
    path = STAGE / name
    assert not path.exists() or name in allowed_existing, name
    assert not name.startswith(('src/bayesian_phystwin/', 'results/', 'evidence/'))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    retained_paths.add(name)
index = STAGE / 'experiments/residual-models'
index.mkdir(parents=True, exist_ok=True)
for family in ['gaussian-processes', 'dirichlet-processes']:
    text = f'# {family.replace("-", " ").title()}\n\n[Research-code index](README.md)\n\nThese are opt-in implementations, not replacements for the current predictor.\n\n| PR | Study | Source files | Paper record |\n| --- | --- | --- | --- |\n'
    for study in manifest['experiments']:
        if study['family'] != family:
            continue
        paths = [i['path'] for i in study['files'] if i['disposition'] == 'research-source' and i['path'].endswith('.py') and '/test' not in i['path']]
        links = '; '.join(f'[`{p}`](../../{p})' for p in paths)
        paper = f'[#{study["paper_pr"]}](https://github.com/FlorianPfaff/BayesianPhysTwin-Paper/pull/{study["paper_pr"]})' if study['paper_pr'] else 'Original source PR; no manuscript promotion'
        text += f'| [#{study["pr"]}](https://github.com/{REPO}/pull/{study["pr"]}) | {study["title"]} | {links} | {paper} |\n'
    name = f'experiments/residual-models/{family}.md'
    (STAGE / name).write_text(text)
    retained_paths.add(name)
for study in manifest['experiments']:
    for item in study['files']:
        if item['disposition'] != 'historical-document-reference' or not item['source_path'].endswith('/README.md'):
            continue
        name = item['source_path']
        if study['pr'] == 962:
            name = name.replace('dp_residual_development_v1', 'dp_residual_kinematic_v1')
        path = STAGE / name
        if path.exists():
            continue
        link = os.path.relpath(index / 'README.md', path.parent)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'# {study["title"]}\n\nResearch-only implementation from PR #{study["pr"]}.\n\n[Maintained code and test index]({link}) · [Original report and execution identity]({item["historical_url"]})\n\nUse the executed source revision for frozen reproduction. The original outcome\nis not a production promotion. No data experiment runs on merge.\n')
        retained_paths.add(name)
name = 'experiments/residual-models/imports-2026-09-07.json'
(STAGE / name).write_text(json.dumps(manifest, indent=2) + '\n')
retained_paths.add(name)
new_python = [n for n in retained_paths if n.endswith('.py')]
subprocess.run(['ruff', 'check', *new_python], check=True)
subprocess.run(['ruff', 'format', '--check', *new_python], check=True)
assert len(list((STAGE / '.github/workflows').glob('*.yml'))) == 90
assert all((STAGE / n).read_bytes() == data for n, data in original.items() if n not in allowed_existing)
env = dict(os.environ, PYTHONPATH=f'{STAGE}:{STAGE / "src"}:{STAGE / "scripts/remote"}', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_residual_experiment_catalog.py', 'tests/test_workflow_inventory_budget.py'], env=env, check=True)
subprocess.run([sys.executable, 'scripts/ci/check_residual_experiments.py', '--output-dir', str(WORK / 'checks')], env=env, check=True)
# Only now publish an entirely new, reviewed staging branch. No existing ref is updated.
assert api('git/ref/heads/main')['object']['sha'] == BASE, 'main advanced: rebuild against its successor'
tree_entries = [{'path': n, 'mode': '100644', 'type': 'blob', 'content': (STAGE / n).read_text()} for n in sorted(retained_paths)]
tree = api('git/trees', {'base_tree': BASE_TREE, 'tree': tree_entries})
commit = api('git/commits', {'message': 'Integrate twelve residual-model research PRs without production promotion\n\nPreserve source-head ancestry, archive all fifteen proposed workflow definitions inactive, separate the colliding DP namespaces, retain exact source custody, and verify the CPU-only suites. Reconcile the inherited 90-file workflow inventory with an explicit v2 infrastructure baseline; preserve v1 unchanged.', 'tree': tree['sha'], 'parents': parents})
created = api('git/refs', {'ref': f'refs/heads/{TARGET}', 'sha': commit['sha']})
assert created['object']['sha'] == commit['sha']
receipt = {'repository': REPO, 'branch': TARGET, 'commit': commit['sha'], 'tree': tree['sha'], 'parents': parents, 'files': len(retained_paths), 'source_hashes_verified': len(expected), 'scientific_data_run': False, 'main_updated': False}
(WORK / 'publication.json').write_text(json.dumps(receipt, indent=2) + '\n')
(WORK / 'source-catalog.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(receipt, indent=2), flush=True)
