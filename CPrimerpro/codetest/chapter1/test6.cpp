#include <iostream>

int main(){
    int sum = 0, value =0;
    std ::cout << "Enter numbers, and press CTRL+Z to stop"<< std ::endl;
    for (;std ::cin >> value;)
    {
        sum += value;
    }
    std :: cout << "Sum is:" << sum << std ::endl;
    return 0;
}