from setuptools import setup

setup(name='referit3d',
      version='0.1',
      description='Comprehension of localizing 3D objects in scenes.',
      url='http://github.com/referit3d/referit3d',
      author='referit3d_team',
      author_email='optas@cs.stanford.edu',
      license='MIT',
      install_requires=['scikit-learn',
                        'matplotlib',
                        'six',
                        'tqdm',
                        'pandas',
                        'plyfile',
                        'requests',
                        'symspellpy',
                        'termcolor',
                        'tensorboardX',
                        'shapely',
                        'pyyaml'
                        ],
      packages=['referit3d'],
      zip_safe=False)
