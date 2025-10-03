#include <iostream>
#include <typeinfo>

int main()
{
    int i=0,&r=i;
    auto a=r;
    const int ci=i,&cr=ci;
    auto b=ci;
    auto c=cr;
    auto d=&i;
    auto e=&ci;
    const auto f=ci;
    auto &g=ci;

    std::cout<<a<<'_'<<b<<'_'<<c<<'_'<<d<<'_'<<e<<'_'<<f<<'_'<<g<<std::endl;
    std::cout<<typeid(a).name()<<'_'<<typeid(b).name()<<'_'<<typeid(c).name()<<'_'<<typeid(d).name()<<'_'<<typeid(e).name()<<'_'<<typeid(f).name()<<'_'<<typeid(g).name()<<std::endl;

    a=42;
    b=42;
    c=42;
    //d=42;
    //e=42;
    //f=42;
    //g=42;
    std::cout<<a<<'_'<<b<<'_'<<c<<'_'<<d<<'_'<<e<<'_'<<f<<'_'<<g<<std::endl;
    std::cout<<sizeof(a)<<'_'<<sizeof(b)<<'_'<<sizeof(c)<<'_'<<sizeof(d)<<'_'<<sizeof(e)<<'_'<<sizeof(f)<<'_'<<sizeof(g)<<std::endl;

    return 0;

}