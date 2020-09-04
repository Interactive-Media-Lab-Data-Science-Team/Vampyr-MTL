import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="MD-MTL",
    version="0.0.7",
    author="Max Jiang",
    author_email="haoyanhy.jiang@mail.utoronto.ca",
    description="An Ensemble Med-Multi-Task Learning Package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Interactive-Media-Lab-Data-Science-Team/Vampyr-MTL",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)