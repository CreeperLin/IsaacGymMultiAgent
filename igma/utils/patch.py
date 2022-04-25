"""Patch for IsaacGymEnvs."""
import sys
import importlib
import importlib.util
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from typing import Optional, Sequence, Union
import types


class IGEImporter(Loader, MetaPathFinder):
    """Patch using import hooks (PEP 302)."""

    def find_spec(
        self, fullname: str, path: Optional[Sequence[Union[bytes, str]]], target: Optional[types.ModuleType] = None
    ) -> Optional[ModuleSpec]:
        """Handle isaacgymenvs imports."""
        if fullname in ['tasks', 'utils']:
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def load_module(self, fullname: str) -> types.ModuleType:
        """Load module by import path."""
        c_name = 'isaacgymenvs.' + fullname
        mod = sys.modules.get(c_name)
        if mod is None:
            mod = importlib.import_module(c_name)
        sys.modules[c_name] = mod
        sys.modules[fullname] = mod
        return mod


sys.meta_path.append(IGEImporter())
