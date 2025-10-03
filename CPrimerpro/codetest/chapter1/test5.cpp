#include <iostream>

int main(){
    int n1,n2;
    std :: cout <<"Please enter 2 intergers:" <<std ::endl;
    std :: cin >> n1 >> n2 ;
    if (n1 > n2)
    {
        std ::cout << "The Larger number is: " << n1 << " and the number between them are: " << std :: endl;
        
        for (int z=n2+1;z<n1 ;z++)
        {
            std:: cout << z << std :: endl;
        }
    }
    else if (n2 > n1)
    {
         std ::cout << "The Larger number is: " << n2 << " and the number between them are: " << std :: endl;
        
        for (int z=n1+1;z<n2 ;z++)
        {
            std:: cout << z << std :: endl;
        }
    }
    else 
    {
        std :: cout << "Both numbers are equal "<< std :: endl;
    }
    return 0;
}