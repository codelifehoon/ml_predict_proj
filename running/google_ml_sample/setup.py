from setuptools import find_packages
from setuptools import setup


required_packages = ['Keras==2.2.4', 'gunicorn==19.9.0', 'pandas==0.23.4', 'sklearn==0.0', 'konlpy==0.5.1','google-cloud-storage==1.13.0']
# dependency_links = ['git+https://github.com/lucasb-eyer/pydensecrf.git']



setup(name='trainer',
      version='0.1',
      packages=['train'],
      install_requires=required_packages,
      # dependency_links=dependency_links,
      include_package_data=True,
      description='description')
