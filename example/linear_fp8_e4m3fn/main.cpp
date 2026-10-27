#include "forward.hpp"

extern "C" int main()
{
    model::init();
    return model::forward();
}
