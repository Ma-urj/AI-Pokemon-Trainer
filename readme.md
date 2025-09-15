# AI Pokemon Trainer

[![MIT License](https://img.shields.io/badge/License-MIT%20License-silver?style=flat-square)](LICENSE)

[![arxiv 2506.23689](https://img.shields.io/badge/Research-2506.23689-B31B1B?logo=arxiv&style=flat-square)](https://arxiv.org/abs/2506.23689)

https://github.com/user-attachments/assets/7f5d9d4b-7ecd-422f-9c22-834708eb996d


## Installation

First, you will need to install some packages, use pip tool to install.

```bash
pip install -r requirements.txt
```

Then you need copy secret_setting.json.example to secret_setting.json, and typing the AI's api information.

## Usage

There are cli tools in the root folder, `cli.py`.

Run `python3 cli.py server`, you will get a web server which started in 8000 port and 18080 port used by websockets, it have beautiful screen. It's goal is for the normal user to watch how AI work.

![](./docs/img.png)

## Technical Details

If you want to know more technical details, you can check the more [detailed technical documentation](https://github.com/Ma-urj/AI-Pokemon-Trainer/blob/main/docs/running_process.md)

We collated the [location of some game memory](https://github.com/Ma-urj/AI-Pokemon-Trainer/blob/main/docs/memory_address.md) based on the information on the Internet

## Reference

[Pokémon Red and Blue/Internal Index Number](https://tcrf.net/Pok%C3%A9mon_Red_and_Blue/Internal_Index_Number)

[Pokémon Red and Blue/RAM map](https://datacrystal.tcrf.net/wiki/Pok%C3%A9mon_Red_and_Blue/RAM_map)
