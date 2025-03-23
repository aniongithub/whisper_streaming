from setuptools import setup, find_packages

setup(
    name='whisper-streaming',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        # Add your dependencies here
    ],
    entry_points={
        'console_scripts': [
            'whisper_online=whisper-streaming.whisper_online:__main__',
            'whisper_online_server=whisper-streaming.whisper_online_server:__main__',
        ],
    },
)
