# Import MultiModalDepDet (doesn't require mamba_ssm)
from .MultiModalDepDet import MultiModalDepDet

# Lazy import for DepMamba (requires mamba_ssm which may not be installed)
# Only import when actually needed
_depmamba = None
def _get_depmamba():
    global _depmamba
    if _depmamba is None:
        try:
            from .DepMamba import DepMamba
            _depmamba = DepMamba
        except ImportError as e:
            raise ImportError(
                "DepMamba requires mamba_ssm package. "
                "Install it with: pip install mamba-ssm\n"
                f"Original error: {e}"
            )
    return _depmamba

# Make DepMamba available but lazy-loaded
def __getattr__(name):
    if name == "DepMamba":
        return _get_depmamba()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
