#include <iostream>
#include "Element/CST.H"
#include "Element/Test.H"

int main(int argc, char **argv)
{

    // This code tests the CST element defined in
    //    src/Element/CST.H
    // using the tests defined in
    //    src/Element/Test.H

    std::cout << "test.element.cst.diractest...";
    try {Element::Test<Element::CST>::Dirac(); std::cout <<"pass"<<std::endl;}
    catch(Util::Exception::UnitTest &e) {std::cout << "failed:" << std::endl << e.what() << std::endl;}

    std::cout << "test.element.cst.sumtounitytest...";
    try {Element::Test<Element::CST>::SumToUnity(); std::cout <<"pass"<<std::endl;}
    catch(Util::Exception::UnitTest &e) {std::cout << "failed:" << std::endl << e.what() << std::endl;}

    std::cout << "test.element.cst.phiderivativetest...";
    try {Element::Test<Element::CST>::PhiDerivative(); std::cout <<"pass"<<std::endl;}
    catch(Util::Exception::UnitTest &e) {std::cout << "failed:" << std::endl << e.what() << std::endl;}


}
