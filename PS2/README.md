FEM Coding Assignment Part 2: CST Shape Functions
=================================================

Checking out the problem repository
-----------------------------------

You can get the code for this assignment either (1) by checking out a fresh copy of the repository:

    git clone https://github.com/solidsgroup/EM525.git --branch ps2

or (2) by simply switching to the `ps2` branch in the copy you already checked out:

    git checkout ps2
    git pull
    
The latter option is encouraged for those who are proficient with git.
As we develop this code, you will need to copy/paste portions of code (and entire files) from previous assignments.
If you keep all your work in the same repository, this can be done easily with `git checkout` and `git difftool`.

Building the code
-----------------

In the EM525 repository, on the `ps2` branch, compile by typing

    make

Execute the code with

    ./bin/main

The output should contain information from failed tests.

    test.element.cst.diractest...failed:
    Dirac test failed: n=0 m=0phi^m(y^n) = 0.000000 != 1.0
    test.element.cst.sumtounitytest...failed:
    SumToUnity test failed: sum=-60539778500895675077088930481106321408.000000 for Y = [0.195465, 0.388121]
    test.element.cst.phiderivativetest...pass

Implementation
--------------

Here you will implement shape functions for a constant strain triangle (CST) element.
Complete the `TODO`s in `src/Element/CST.H`.
You will complete the code for the shape function `phi` that calculates 
```math
\begin{bmatrix} \phi_1(\mathbf{Y}) & \phi_2(\mathbf{Y}) & \phi_3(\mathbf{Y}) \end{bmatrix} 
```
and a function `Dphi` that calculates the derivatives
```math
\begin{bmatrix} \frac{\partial\phi_1}{\partial\mathbf{Y}} & \frac{\partial\phi_2}{\partial\mathbf{Y}} & \frac{\partial\phi_2}{\partial\mathbf{Y}}\end{bmatrix} 
```

Testing
-------

Test the code by compiling with `make` and running with `./bin/main`.
The code runs three tests:

1. `DiracTest`: ensure that shape function "n" is one at "n" and zero at all other points.
    
2. `SumToUnityTest`: ensure that the shape functions sum to unity at points randomly selected in the element.
  
3. `PhiDerivativeTest`: ensure that the exact derivatives match the numerically calculated derivatives.

Be sure to submit your implementation and the output of your code.
