from setuptools import setup, find_packages

# read the contents of the README.md file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md')) as f:
	long_description = f.read()

setup(name='hcipy',
	version='0.2',
	description='A framework for performing optical propagation simulations, meant for high contrast imaging, in Python.',
	long_description=long_description,
	long_description_content_type='text/markdown',
	url='https://gitlab.strw.leidenuniv.nl/por/hcipy',
	author='Emiel Por',
	author_email='por@strw.leidenuniv.nl',
	packages=find_packages(),
	install_requires=[
		"numpy",
		"scipy",
		"matplotlib>=2.0.0",
		"Pillow",
		"pyyaml",
		"mpmath",
		"astropy",
		"imageio"],
	zip_safe=False,
	classifiers=(
		"Development Status :: 3 - Alpha",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
		"Programming Language :: Python :: 2",
		"Programming Language :: Python :: 3",
		"Topic :: Scientific/Engineering :: Astronomy"
	)
)