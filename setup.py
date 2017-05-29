from setuptools import setup

with open('README.rst') as file:
    long_description = file.read()

setup(
    name='mir_eval',
    version='0.4',
    description='Common metrics for common audio/music processing tasks.',
    author='Colin Raffel',
    author_email='craffel@gmail.com',
    url='https://github.com/craffel/mir_eval',
    packages=['mir_eval'],
    long_description=long_description,
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        'Development Status :: 5 - Production/Stable',
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
    ],
    keywords='audio music mir dsp',
    license='MIT',
    install_requires=[
        'numpy >= 1.7.0',
        'scipy >= 0.14.0',
        'future',
        'six'
    ],
    extras_require={
        'display': ['matplotlib>=1.5.0',
                    'scipy>=0.16.0'],
        'testing': ['matplotlib>=2.0.0']
    },
    entry_points = {
        'console_scripts':
        ['beat_eval=evaluators.beat_eval:main',
         'chord_eval=evaluators.chord_eval:main',
         'melody_eval=evaluators.melody_eval:main',
         'multipitch_eval=evaluators.multipitch_eval:main',
         'onset_eval=evaluators.onset_eval:main',
         'pattern_eval=evaluators.pattern_eval:main',
         'segment_eval=evaluators.segment_eval:main',
         'segment_hier_eval=evaluators.segment_hier_eval:main',
         'separation_eval=evaluators.separation_eval:main',
         'tempo_eval=evaluators.tempo_eval:main',
         'transcription_eval=evaluators.transcription_eval:main']
    }
)
