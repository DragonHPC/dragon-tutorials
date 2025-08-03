# Getting Started with Dragon

Dragon requires a Linux environment in which to run programs. This tutorial shows you how to get the proper environment configured to run on your laptop as a first step to discovering Dragon's capabilities. Dragon is designed from the ground-up to run multinode on a cluster or supercomputer, but it can also run on your laptop. From laptop to exaflop is a Dragon goal.

If you have a Linux laptop or Linux server laying around then you may be able to download Python, create a Python virtual environment, pip install dragon, and you might be off to the races.

For those of us that don't have a Linux laptop or Linux server at our disposal, setting up an environment on a Windows laptop or a Macbook is not so hard. You will what to download [VS Code](http://code.visualstudio.com/) and [Docker Desktop](http://www.docker.com/). Download both packages, install them, and then follow the directions below for your particular laptop environment.

The [test program](test.py) that is run later in this tutorial is provided. You can copy and paste it when you are ready to try it out.

## Installing on Windows

On Microsoft Windows make sure you include/activate the WSL2 when installing Docker Desktop. The WSL2 (Windows Subsystem for Linux 2) provides the underlying functionality you need to run Linux containers on Windows. A container is like a lightweight virtual machine for your Laptop.

You can follow the directions provided in this [video of setting up Dragon for Windows laptops](https://youtu.be/Cn6uQ76DdQ8).

The setup involves downloading the [Homework repository](https://github.com/kentdlee/homework) which contains code that VS Code will use to automatically start a container for you. Just click on the Code button and select to download the zip file.

Once downloaded, extract the zip file. Then open the resulting homework-main folder in VS Code. After opening the folder VS Code will ask if you want to restart it in a container. Click on that. It may take a bit the first time for that container to be built and started. Once it starts, click on Terminal in VS Code and the + button will start a new terminal. In this terminal you can create a Python virtual environment and activate it with these commands.

```bash
python3 -m venv _env
. _env/bin/activate
```

Then you can install Dragon in the virtual environment you just created.
```bash
pip install dragonhpc
```

Once that is installed you are ready to try out the [test program](test.py). If it runs successfully, congratulations! You have created an environment in which you can run Dragon programs.

If you come back to use Dragon at a later time, you can follow the directions to open the folder in VS Code, start a terminal, and then execute

```bash
. _env/bin/activate
```

to reactivate the Python virtual environment. You do not need to pip install again when you come back to use Dragon later. If you mess things up by, just follow the steps above again to create a different container.

## Installing on Macbooks with Intel Chips

On older Macbooks you can follow the steps for installing on Windows except that there is no equivalent of a WSL2 to include in installation and no need for anything else. You just need to install VS Code and Docker Desktop and follow the rest of the instructions above.

## Building Dragon (currently required for Apple Silicon Macbooks)

Soon (I am writing this on 8/2/2025.) you will not need to follow these directions to run Dragon on your laptop, but as of this writing you still do. Windows users and older Intel Macbook users can also follow these directions. Following these directions will build the Dragon source code from scratch and give you access to all the Dragon source code as well as examples and test programs used to test Dragon which give you snippets of code examples that may be helpful.

You still need VS Code and Docker Desktop, so make sure you download and install these two packages.

You follow the directions provided in this [video of building Dragon from source](https://youtu.be/jgeHWb5V00k).

This setup involves downloading the [Dragon Source Code](https://github.com/DragonHPC/dragon). which contains code that VS Code will use to automatically start a container for you. Just click on the Code button and select to download the zip file.

Once download, extract the zip file. Once downloaded, extract the zip file. Then open the resulting dragon-main folder in VS Code. After opening the folder VS Code will ask if you want to restart it in a container. Click on that. It may take a bit the first time for that container to be built and started. Once it starts, click on Terminal in VS Code and the + button will start a new terminal. In this terminal you will do a clean build of Dragon.

```bash
. hack/clean_build
```

Doing the build in this way will create a new Python virtual environment and activate it. Then it will build dependencies and then build the Dragon code. Once the build is complete you have a Dragon environment in which you can run Dragon programs and you are ready to try out the [test program](test.py). If it runs successfully, congratulations! You have built Dragon from source and you have an environment in which you can run Dragon programs.

If you come back to use Dragon at a later time, you can follow the directions to open the folder in VS Code, start a terminal, and then execute

```bash
. _env/bin/activate
```

to reactivate the Python virtual environment. You do not need to build Dragon again when you come back to use Dragon later. If you mess things up by, just follow the steps above again to create a different container and build Dragon in it.