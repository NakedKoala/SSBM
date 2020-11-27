from setuptools import setup, find_packages
setup(
    name='cs486-ssbm-bot',
    version='0.0.1',
    description="SSBM Bot using supervised training and reinforcement learning for the CS 486 AI project.",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    # ext_modules=extensions,
    # cmdclass=cmdclass,
    packages=find_packages(),
    # entry_points=entry_points,
    install_requires=[
        'torch>=1.6.0',
        'numpy',
        'pandas',
        'scikit-learn',
        'tqdm',
        'py-slippi',
        'melee',
    ],
    package_data={},
    # url='',
    # download_url='',
    # author='',
    # author_email='',
    # python_requires='>={}'.format(python_min_version_str),
    # PyPI package information.
    classifiers=[]
    # license='BSD-3',
    # keywords='',
)
