#include <iostream>

#include "Set/Set.H"

int main(int argc, char **argv)
{
    // TIP: We will use the "Eigen" library to do matrix and vector manipulations.
    //      You can find documentation about Eigen on its webpage
    //            https://eigen.tuxfamily.org/

    // TODO #1: Print "Hello world" to the console by uncommenting the following line:
    //
    std::cout << "Hello world" << std::endl;
        

    // TODO #2: Create a vector "u" with components [1,2] by uncommenting
    //          the following line:
    // 
    Set::Vector u(1.0,2.0);


    // TODO #3: Create another vector "v" with components [3,4]
    Set::Vector v(3.0,4.0);

    
    // TODO #4: Create a zero-filled matrix A by uncommenting the following lines:
    //
    Set::Matrix A = Set::Matrix::Zero();    

    
    // TODO #5: Compute the product Au and print the output:
    Set::Vector w = A*u;
    std::cout << w <<std::endl;

    
    // TODO #6: Compute the dot product of u and v and store in a scalar variable by
    //          uncommmenting the following lines.
    //
    Set::Scalar udotv = u.transpose()*v;
    std::cout << udotv << std::endl;



    // TODO #7: Compute the dyadic product of u and v and store in a marrix called "D"
    Set::Matrix D = u*v.transpose();
    std::cout << "The dyadic (outer) product is: \n" << D << std::endl;

    // TODO #8: Compute the determinant of D and print to the screen.
    Set::Scalar detD = D.determinant();
    std::cout << "The determinant of D = \n" << detD << std::endl;

    return 0;
}
