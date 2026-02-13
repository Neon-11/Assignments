#ifndef ELEMENT_ELEMENT_H
#define ELEMENT_ELEMENT_H

#include "Set/Set.H"

namespace Element
{
template<int D, int N, int Q>
class Element
{
public:

    Element()
    {}
                
    Element(std::vector<Set::Vector> &a_X0, std::array<int,N> a_id) :
        X0(&a_X0), id(a_id)
    {}

    Element(std::array<Set::Vector,N> &a_X0) 
    {
        X0 = new std::vector<Set::Vector>();
        for (int n = 0 ; n < N ; n++)
        {
            X0->push_back(a_X0[n]);
            id[n] = n;
        }
    }

    virtual std::array<double,N> phi(Set::Vector y) = 0;
    virtual std::array<Set::Vector,N> Dphi(Set::Vector y) = 0;

public:
    constexpr int getN() {return N;}
    virtual const std::array<Set::Vector,N> & getY0() = 0;

    static const int _D = D; // The DIMENSION of the space (2d or 3d)
    static const int _N = N; // The number of NODES in the element
    static const int _Q = Q; // The number of QUADRATURE POINTS in the element

private:
    std::vector<Set::Vector> *X0;
    std::array<int,N> id;
};



}
#endif
