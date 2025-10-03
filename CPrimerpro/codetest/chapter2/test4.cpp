#include <iostream>
#include <typeinfo>

int main()
{
    const int i=42;
    auto j=i;
    const auto &k=i;
    auto *p=&i;
    const auto j2=i,&k2=i;
    std::cout<<typeid(j).name()<<'_'<<typeid(k).name()<<'_'<<typeid(p).name()<<'_'<<typeid(j2).name()<<'_'<<typeid(k2).name()<<std::endl;

    return 0;

}