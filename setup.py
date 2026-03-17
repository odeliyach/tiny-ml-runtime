from setuptools import setup, Extension

tinymlinference = Extension(
    "tinymlinference",
    sources=["src/c/inference_module.c"],
    extra_compile_args=["-O2", "-Wall"],
    libraries=["m"],
)

with open("README.md") as f:
    long_description = f.read()

setup(
    name="tinymlinference",
    version="1.0.0",
    description="Generic neural network inference engine in pure C — callable from Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Odeliya Charitonova",
    url="https://github.com/odeliyach/tiny-ml-runtime",
    ext_modules=[tinymlinference],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: C",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
    ],
)
