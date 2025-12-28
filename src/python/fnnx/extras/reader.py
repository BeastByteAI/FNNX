import tarfile
import json
import re

from fnnx.extras.pydantic_models.manifest import Manifest
from fnnx.extras.pydantic_models.meta import MetaEntry
from fnnx.extras.pydantic_models.envs import Python3_CondaPip
from fnnx.extras.jsonpatcher import apply_patches


class Reader:
    def __init__(self, model_path: str):
        with tarfile.open(model_path, "r:*") as tar:
            self.manifest = Manifest(**self._load_manifest(tar))
            self.metadata = self._load_metadata(tar)
            self.env = json.loads(self._get_file(tar, "env.json"))

        self.pyenv = None
        if "python3::conda_pip" in self.env:
            self.pyenv = Python3_CondaPip(**self.env["python3::conda_pip"])

    def _load_manifest(self, tar: tarfile.TarFile):
        manifest_data = json.loads(self._get_file(tar, "manifest.json"))

        patch_pattern = re.compile(r"^manifest-[^/]+\.patch\.json$")
        patch_files = [
            member.name
            for member in tar.getmembers()
            if patch_pattern.match(member.name)
        ]
        if patch_files:
            patches = [json.loads(self._get_file(tar, pf)) for pf in patch_files]
            manifest_data = apply_patches(manifest_data, patches)

        return manifest_data

    def _load_metadata(self, tar: tarfile.TarFile):
        # Match only "meta.json" or "meta-{uid}.json" at root level
        meta_pattern = re.compile(r"^meta(-[^/]+)?\.json$")
        entries = []
        for member in tar.getmembers():
            if meta_pattern.match(member.name):
                entries.extend(json.loads(self._get_file(tar, member.name)))
        return [MetaEntry(**m) for m in entries]

    def _get_file(self, tar: tarfile.TarFile, target: str):
        member = tar.getmember(target)
        f = tar.extractfile(member)
        if not f:
            raise ValueError(f"Could not locate `{target}`")
        return f.read().decode("utf-8")
