FEM Coding Assignment Part 1: Introduction to linux and C++
===========================================================

In this course we will develop a finite element analysis code in C++. 
You are not expected to know C++ yet, and so we will have a number of coding exercises and intermediate steps to help you get familiar with it. 
The purpose of this assignment is to prepare you to use basic computing resources for code development.


Setting up a linux environment
------------------------------

It is recommended that you work in a linux environment. 
Though not required, this is the only OS that will be supported in this class. 
There are several ways you can gain access to a linux environment:
*  The engineering college maintains a number of linux remote computing options. 
   These are available here: https://it.engineering.iastate.edu/remote-computing/
*  The Windows Subsystem for Linux (WSL) allows users to run linux locally within a Windows environment. 
   You can find instructions here: https://learn.microsoft.com/en-us/windows/wsl/install
*  You can install linux locally using VirtualBox: https://www.virtualbox.org/
*  You can install linux to dual-boot on most computers.


Checking out the problem repository
-----------------------------------

Once you have your linux environment set up, check out this github repository.
You can do this by running the command:

    git clone https://github.com/solidsgroup/EM525.git --branch ps1

which will create a folder titled EM525 on your computer.
Note: if you do not have git installed, you will need to install it. 
On Ubuntu, you can install git with the command

    sudo apt install git

Once you have checked out the repository, change to that directory with

    cd EM525

Installing Eigen
----------------

You can skip this command if you have eigen already installed on your system.
If you do not have it installed, you can simply run

    make eigen

to automatically clone a local copy.

Building and running the code
-----------------------------

In the EM525 directory, type

    make
   
This generates an executable in a directory called bin.
You can run this exectuable by 

    ./bin/main

If it runs correctly, there will be no output. 


Completing the assignment
-------------------------

Open the file `./src/main.cpp` in a text editor (such as: emacs, vim, vscode, etc.).
You will see a number of `TODO` comments that outline what is required to complete the assignment.
Once you have implemented these changes, compile the code again by typing

    make
    
Make sure that it compiles without errors. 
**You must recompile every time you change the code**.
Once it has done so, run the code with

    ./bin/main


