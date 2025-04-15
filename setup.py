from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='openagentkit',
    version='0.1.0.dev6',
    packages=find_packages(),
    install_requires=requirements,
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
