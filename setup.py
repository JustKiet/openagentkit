import os
from setuptools import setup, find_packages
from itertools import chain

here = os.path.abspath(os.path.dirname(__file__))

def load_requirements(filename: str):
    with open(os.path.join(here, 'requirements', filename)) as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith('#')
        ]

requirements_core = load_requirements("requirements-core.txt")
requirements_openai = load_requirements("requirements-openai.txt")
requirements_milvus = load_requirements("requirements-milvus.txt")
requirements_voyageai = load_requirements("requirements-voyageai.txt")
requirements_redis = load_requirements("requirements-redis.txt")
requirements_valkey = load_requirements("requirements-valkey.txt")
requirements_dev = load_requirements("requirements-dev.txt")

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

requirements_all = sorted(set(chain(
    requirements_core,
    requirements_openai,
    requirements_milvus,
    requirements_voyageai
)))

setup(
    name='openagentkit',
    version='0.1.0-alpha.4',
    packages=find_packages(),
    install_requires=requirements_core,
    extras_require={
        'openai': requirements_openai,
        'milvus': requirements_milvus,
        'voyageai': requirements_voyageai,
        'redis': requirements_redis,
        'valkey': requirements_valkey,
        'dev': requirements_dev,
        'all': requirements_all,
    },
    include_package_data=True,
    author='Kiet Do',
    author_email='kietdohuu@gmail.com',
    description='An open-source framework for building and deploying AI agents.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='Apache-2.0',
    keywords='AI, agents, open-source, llm, tools, executors',
    python_requires='>=3.11',
    url='https://github.com/JustKiet/openagentkit',
    project_urls={
        'Bug Reports': 'https://github.com/JustKiet/openagentkit/issues',
        'Source': 'https://github.com/JustKiet/openagentkit',
        'Documentation': 'https://github.com/JustKiet/openagentkit#readme',
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.12',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
)
