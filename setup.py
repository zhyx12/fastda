import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
  name="fastda",
  version="0.0.13",
  author="Yixin Zhang",
  author_email="zhyx12@mail.ustc.edu.cn",
  description="A simple framework for unsupervised domain adaptation",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/zhyx12/fastda",
  packages=setuptools.find_packages(),
  classifiers=[
  "Programming Language :: Python :: 3",
  "License :: OSI Approved :: MIT License",
  "Operating System :: OS Independent",
  ],
)