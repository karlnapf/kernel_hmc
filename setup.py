from setuptools import setup, find_packages


setup(name='kernel_hmc',
      version='0.1',
      description='Code for NIPS 2015 Gradient-Free Hamiltonian Monte Carlo with Efficient Kernel Exponential Families',
      url='https://github.com/karlnapf/kernel_hmc',
      author='Heiko Strathmann',
      author_email='heiko.strathmann@gmail.com',
      license='BSD3',
      packages=find_packages('.', exclude=["*tests*", "*.develop"]),
      zip_safe=False)
