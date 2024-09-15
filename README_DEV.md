1) Info
https://docs.epam-rail.com/Cookbook/dial-cookbook/examples/how_to_call_text_to_text_applications

2) Dev environment
a) Setup
```sh
curl -sSL https://install.python-poetry.org | python3 - --version 1.7.0
poetry init
poetry config virtualenvs.in-project true
poetry env use 3.11
source .venv/bin/activate
poetry config virtualenvs.prompt 'ai-dial'
poetry config --list
```

b) Install
```sh
poetry install --no-root 
```

2) Run app in docker desktop
```sh
cd dial-docker-compose/application && docker-compose up -d --build --force-recreate
```
-d : Detached mode, runs containers in the background.\
--build : Builds images before starting containers.\
--force-recreate : Recreates containers even if their configuration and image haven't changed.\

3) Submodule
a) Add submodule
```
git submodule init
git submodule update
```
Submodule 'dial-sdk' (https://github.com/epam/ai-dial-sdk) registered for path '../../dial-sdk'
Cloning into '/home/he/Projects/Prod/EPAM/ai-dial/dial-sdk'...
Submodule path '../../dial-sdk': checked out 'a17a4cf7c1493b7f0a7fe45e780764d088222891'

b) Delete submodule (Use carefully)
```sh
git submodule deinit -f dial-sdk
rm -rf dial-sdk
git rm -f dial-sdk
git add .gitmodules
git commit -m "Remove submodule"
```

4) Subtree (preferable in case of fork changes, to do not push to remote branch, but use core ai-dial project commits)
a) Add Subtree
```sh
git subtree add --prefix=dial-sdk https://github.com/epam/ai-dial-sdk development
ls dial-sdk
```
b) Push to subtree
```sh
git subtree push --prefix=path/to/subtree https://github.com/epam/ai-dial-sdk development
```
c) Pull from subtree
```sh
git subtree pull --prefix=path/to/subtree https://github.com/epam/ai-dial-sdk development
```

5) Clean git cached files
```sh
git rm --cached -r .idea/
git rm --cached -r .vscode/
```