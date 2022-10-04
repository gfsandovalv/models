from setuptools import setup
import versioneer

requirements = [
    # package requirements go here
]

setup(
    name='models',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="A python module for fitting and modeling.",
    license="MIT",
    author="Gabriel Sandoval",
    author_email='gfsandovalv@unal.edu.co',
    url='https://github.com/gfsandovalv/models',
    packages=['models'],
    entry_points={
        'console_scripts': [
            'models=models.cli:cli'
        ]
    },
    install_requires=requirements,
    keywords='models',
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ]
)
