"""Passive installed-code identity and readiness; never an execution certificate."""
from solverpilot._identity import source_tree_sha256
from importlib import metadata
from importlib.util import find_spec
import json
from pathlib import Path
import platform


def readiness_report():
    import solverpilot
    root=Path(solverpilot.__file__).resolve().parent
    code_id=source_tree_sha256()
    distribution_version=None; matches=False; editable=False
    try:
        dist=metadata.distribution('solverpilot')
        distribution_version=dist.version
        matches=any(str(p).replace('\\','/')=='solverpilot/__init__.py'
                    and Path(dist.locate_file(p)).resolve()==root/'__init__.py' for p in (dist.files or []))
        direct=json.loads(dist.read_text('direct_url.json') or '{}')
        editable=bool(direct.get('dir_info',{}).get('editable',False))
    except metadata.PackageNotFoundError:
        pass
    except (ValueError,OSError):
        # Malformed installation metadata cannot establish import ownership.
        matches=False
    def installed(name):
        try:return find_spec(name) is not None
        except (ImportError,ValueError):return False
    dependencies={name:installed(name) for name in ('scipy','highspy','osqp','pyscipopt','clarabel','casadi','ortools','cuopt')}
    capabilities=[
        {'name':'core_lp_milp_convex_qp','maturity':'stable_api','included_in_published_0_4':True},
        {'name':'exact_scip_vipr','maturity':'explicit_optional','included_in_published_0_4':False,
         'requires':'caller-qualified SCIP exact and VIPR executable paths'},
        {'name':'cuopt_gpu_lp','maturity':'experimental','included_in_published_0_4':False,
         'requires':'Linux/WSL2, CUDA, compatible device and cuOpt'},
        {'name':'isolated_cpu_deadline','maturity':'development','included_in_published_0_4':False},
        {'name':'extended_equality_recovery','maturity':'development','included_in_published_0_4':False},
        {'name':'guarded_learned_lp','maturity':'experimental','included_in_published_0_4':False},
    ]
    return {'schema':'solverpilot.readiness.v1','package_version':solverpilot.__version__,
            'code_lineage':'development after published 0.4','source_sha256':code_id,
            'distribution_version':distribution_version,'distribution_owns_import':matches,
            'editable_install_metadata':editable,'import_path':str(root),
            'platform':platform.platform(),'dependency_modules_found':dependencies,
            'capabilities':capabilities,'execution_qualified':False,
            'automatic_learned_routing_enabled':False,'automatic_gpu_routing_enabled':False,
            'qualification_note':'Module presence and code identity do not establish solver, hardware or performance qualification.'}
