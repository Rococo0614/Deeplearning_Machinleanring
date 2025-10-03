#include <iostream>

int main()
{
    int n=50,sum=0;
    while (n <= 100)
    {
        sum+=n;
        n++;
    }
    std::cout << "The sum of nuumbers from 50 to 100 is " << sum <<std :: endl;
    return 0;
}