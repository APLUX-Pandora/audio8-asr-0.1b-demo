"""Install the paired prebuilt library; never compile the model engine at pip install."""
import hashlib
import json
from pathlib import Path
import shutil
from setuptools import Distribution, setup
from setuptools.command.build_py import build_py
from setuptools.errors import SetupError
from wheel.bdist_wheel import bdist_wheel

ROOT = Path(__file__).resolve().parent


class BuildSDK(build_py):
    def run(self):
        pairing = json.loads((ROOT / "native-library.json").read_text())
        library = ROOT / "lib" / pairing["filename"]
        if not library.is_file():
            raise SetupError("Missing matched lib/libaudio8.so; obtain the SDK Release before installing")
        payload = library.read_bytes()
        if payload[:4] != b"\x7fELF" or int.from_bytes(payload[18:20], "little") != 183:
            raise SetupError("The SDK payload must be a Linux AArch64 ELF")
        if hashlib.sha256(payload).hexdigest() != pairing["sha256"]:
            raise SetupError("Native SDK SHA256 does not match native-library.json")
        notices = ("LICENSE", "THIRD_PARTY_NOTICES.txt", "LICENSES")
        for name in notices:
            if not (ROOT / "lib" / name).exists():
                raise SetupError("Missing native SDK licensing: " + name)
        bundled_qnn = ROOT / "lib/qnn248"
        if not (bundled_qnn / "inventory.json").is_file():
            raise SetupError("Missing bundled QNN runtime")
        inventory = json.loads((bundled_qnn / "inventory.json").read_text())
        expected = {entry["path"]: entry["sha256"] for entry in inventory["files"]}
        actual = {p.relative_to(bundled_qnn).as_posix() for p in bundled_qnn.rglob("*.so") if p.is_file()}
        if not expected or len(expected) != len(inventory["files"]) or actual != expected.keys():
            raise SetupError("Bundled QNN libraries do not match inventory: " + ", ".join(sorted(actual ^ expected.keys())))
        for path, digest in expected.items():
            if hashlib.sha256((bundled_qnn / path).read_bytes()).hexdigest() != digest:
                raise SetupError("Bundled QNN library SHA256 does not match inventory: " + path)
        # Rebuild package directories so removed modules cannot survive a cached build.
        build_dir = Path(self.build_lib)
        for package in ("audio8", "pandora_audio8_asr_0_1b", "audio8_runtime"):
            generated = build_dir / package
            if generated.exists():
                shutil.rmtree(generated)
        for module in ("audio8_types", "audio8_audio", "audio8_hardware", "audio8_package", "_audio8_native"):
            (build_dir / (module + ".py")).unlink(missing_ok=True)
        super().run()
        target = build_dir / "pandora_audio8_asr_0_1b" / "_lib"
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(library, target / library.name)
        shutil.copytree(bundled_qnn, target / "qnn248")
        shutil.copy2(ROOT / "native-library.json", target / "native-library.json")
        (target / "docs").mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "RUNTIME_API.md", target / "docs/RUNTIME_API.md")
        assets = ROOT / "assets"
        (target / "assets").mkdir(parents=True, exist_ok=True)
        for name in ("silero_vad.onnx", "Silero-VAD-MIT.txt"):
            shutil.copy2(assets / name, target / "assets" / name)
        for name in notices:
            source = ROOT / "lib" / name
            if source.is_dir():
                shutil.copytree(source, target / name, dirs_exist_ok=True)
            else:
                shutil.copy2(source, target / name)


class BinarySDKDistribution(Distribution):
    def has_ext_modules(self):
        # Install the prebuilt SDK into platlib at the wheel root.
        return True


class LinuxAArch64Wheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.plat_name = "linux_aarch64"

    def get_tag(self):
        return "py3", "none", "linux_aarch64"


setup(distclass=BinarySDKDistribution, cmdclass={"build_py": BuildSDK, "bdist_wheel": LinuxAArch64Wheel})
