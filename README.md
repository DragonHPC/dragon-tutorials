# Welcome to Dragon Tutorials

Welcome! The purpose of this site is to provide tutorials and how-tos related to Dragon. Discussions on this site are also open to you to ask questions, suggest topics, and offer your own experiences with Dragon! No questions are bad questions as long as they are asked sincerely. Please don't hesitate to speak up if you have questions.

Dragon is unlike most libraries in that it is designed not to replace anything, but to make everything work better together and to remove the burden of rolling your own internode communication and synchronization when writing parallel programs in a distributed computing environment.

![Conceptual View](images/conceptual.png)

Conceptually, Dragon provides you with a unified view of your distributed computation. The same version of Dragon runs on your laptop and supercomputers. The same version of your program can run on a supercomputer and your laptop (though at a much smaller scale, of course)! You can also optimize your program to run the best on your particular hardware. Dragon is very flexible and is a great way to prototype a solution and then optimize the heck out of it once you have something running.

Dragon is a composable, distributed runtime that enables users to create sophisticated, scalable, resilient, and high-performance AI/HPC applications, workflows, and services via Python, C, C++, and potentially anything that can be built and execute in one of those three environments.

Dragon provides a run-time that manages off-node communication using a design that is exactly the same for on-node communication. The only difference to your program between on-node communication and off-node communication is a performance
difference since off-node communication occurs over a high-speed network and on-node communication utilizes shared memory. The cost of scaling past what a node can handle is off-node communication, and Dragon handles all this for you in a very efficient manner using a transport agent to do the data movement between nodes.

When you start a Dragon program, you run the `dragon` command which brings up the Dragon runtime and executes your program. Running a Dragon
program looks like this.

```bash
    dragon prog.py
```

Where prog.py is a Python program or executable. Most of the time you would start your program via a Python program. The `dragon` command brings up the Dragon run-time on all your nodes, ties them all together using the Dragon transport agents, and starts your program on one of the nodes. That program can then expand to all the nodes on your distributed computer by starting other processes as part of its parallel program.

![Dragon Architecture](images/dragonarch.png)

While your program gets started on a particular node by Dragon, it still conceptually has access to all the nodes of the distributed computer using what is called Python multiprocessing or by using Dragon APIs which all operate exactly the same whether your application is running single node or multinode.

The purpose of this site then is to get you up and running Dragon and writing Dragon programs as quickly as possible. And, it will introduce you to a lot of what Dragon can do by providing you with tutorials and examples. The first tutorials show you how to set up a development environment in which you can write Dragon programs. After that we'll dive deeper into Dragon. Each tutorial will also provide an `In Depth` section that will further explain how it all works to give interested readers a better understanding of Dragon from top to bottom.

For complete Dragon documentation, please see http://dragonhpc.org. That site contains API documentation, additional examples, links to the Dragon source code repository, and other discussions.

# Getting Started

TBD.