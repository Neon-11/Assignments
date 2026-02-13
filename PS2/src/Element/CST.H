#ifndef ELEMENT_CST_H
#define ELEMENT_CST_H

#include "Element.H"

namespace Element
{
class CST : public Element<2,3,1>
{
public:
    using Element<2,3,1>::Element;


    /**
    *  Constant Strain Triangle Element (CST)
    * 
    *        Y2
    *        |
    *  (0,1) O
    *        |\
    *        |  \
    *        |    \
    *        |      \
    *        |        \
    *        |          \
    *        |            \
    *  (0,0) O-------------O---  Y1
    *                     (1,0)
    * 
    *  The three shape functions are
    * 
    *      phi1 = 1 - Y1 - Y2
    *      phi2 = Y1
    *      phi3 = Y2
    */
    

    std::array<double,3> phi(Set::Vector Y) override
    {
        // 
        // TODO #1
        // 
        // Write the code to return an array of values, such that
        // each value contains the value of the corresponding
        // shape function:
        //
        // input: Y
        //
        // output: [phi_0(Y), phi_1(Y), phi_2(Y)]
        //
        // Notes:
        //   - C++ indexes from 0; e.g Y(0) is the x component,
        //     and Y(1) is the y component, etc.
        //   - you can access the ith component of the
        //     retrn value with ret[i]
        //   - you can access the ith component of the
        //     vector with Y(i)
        // 

        std::array<double,3> ret; // <-- store the return values in this variable

        // ** your code goes here **
	ret[0] = 1.0 - Y[0] - Y[1];
	ret[1] = Y[0];
	ret[2] = Y[1];

        return ret;               // <-- they are returned here
    }

    std::array<Set::Vector,3> Dphi(Set::Vector Y) override
    {

        // 
        // TODO #2
        // 
        // Write the code to return the derivatives of the shape 
        // functions.
        //
        // input: Y 
        //
        // output: [{dphi_0/dY_0, dphi_0/dY_1},
        //          {dphi_1/dY_0, dphi_1/dY_1},
        //          {dphi_2/dY_0, dphi_2/dY_1}]
        //

        std::array<Set::Vector,3> ret; // <-- store the return values here

        // ** your code goes here **
	ret[0] = {-1, -1};
	ret[1] = {1, 0};
	ret[2] = {0, 1};

        return ret;
    }

public:
    //
    // This is a utility function 
    //
    virtual const std::array<Set::Vector,3> & getY0() {return Y0;}

private:
    //
    // Hard-coded values common to all elements
    //

    const std::array<Set::Vector,3> Y0 = {
        Set::Vector(0.0,0.0),
        Set::Vector(1.0,0.0),
        Set::Vector(0.0,1.0)
    };

};

}
#endif
