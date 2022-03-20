from setuptools import setup, find_packages

setup(name='traj_pred',
      version='1.0.0',
      description='Ball modeling and trajectory forecasting',
      url='https://github.com/sebasutp/trajectory_forcasting',
      author='Sebastian Gomez-Gonzalez',
      author_email='sgomez@tue.mpg.de',
      packages=find_packages(),
      setup_requires=['pytest-runner'],
      tests_require=['pytest'],
      install_requires = [
        'numpy',
        'matplotlib',
        'pyzmq',
        'keras'
        ],
      license='Do not know')
