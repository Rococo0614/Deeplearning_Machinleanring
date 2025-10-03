#include <iostream>
#include "Sales_item.h"

int main()
{
    Sales_item trans, total;
    std :: cout << "Enter the book information "<< std ::endl;
    if (std :: cin >> total )
    {
        while (std :: cin >> trans )
        {
            if (compareIsbn(total, trans))
            total += trans;
            else 
            {
                std :: cout << total << std ::endl;
                total = trans ;
            }
        }
        std :: cout << total <<std ::endl;
    }
    else 
    {
        std :: cerr << "No data?!" << std :: endl;
        return -1;
    }
    return 0;

}