import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tacotron2-model",
    version="0.2.2",
    author="Ben Andrew",
    author_email="benandrew89@gmail.com",
    description="A PyPI port of the NVIDIA Tacotron2 model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BenAAndrew/tacotron2-model",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    install_requires=[],
    python_requires=">=3.5",
)
