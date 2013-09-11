from setuptools import setup

setup(
    name='mir_eval',
    version='0.0.1',
    description='Common accuracy scores for common audio/music processing tasks.',
    author='Colin Raffel',
    author_email='craffel@gmail.com',
    url='https://github.com/craffel/mir_eval',
    packages=['mir_eval'],
    long_description="""\
    Collection of Python scripts to compute common heuristic accuracy scores for various music/audio information retrieval/signal processing tasks.
    """,
    classifiers=[
          "License :: OSI Approved :: GNU General Public License (GPL)",
          "Programming Language :: Python",
          "Development Status :: 3 - Alpha",
          "Intended Audience :: Developers",
          "Topic :: Multimedia :: Sound/Audio :: Analysis",
    ],
    keywords='audio music mir dsp',
    license='GPL',
    install_requires=[
        'numpy >= 1.7.0',
        'scipy',
        'scikit-learn',
    ],
)
