try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name='EZPZLEMONSQZTrack',
      description='Linear Track Tracker for Matteo',
      author='Ronny Eichler',
      author_email='r.eichler@science.ru.nl',
      version='0.1.0',
      install_requires=['tqdm', 'numpy', 'scipy', 'opencv-contrib-python', 'matplotlib'],
      packages=['ezpzlemonsqz'],
      entry_points="""[console_scripts]
            ezpztrack=ezpzlemonsqz.track:main""")
