from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    name="snakefusion",
    version="0.1.0",
    rust_extensions=[RustExtension("snakefusion._snakefusion", binding=Binding.PyO3)],
    packages=["snakefusion"],
    # rust extensions are not zip safe, just like C-extensions.
    zip_safe=False,
)
