#include <print>
#include <vector>
#include <variant>

struct Leaf  {
    double residual;
};

struct Decision {
    std::string indicator;
    double threshold;
};

namespace EnumResolve {
    template<typename... Ts>
    struct overloaded: Ts... { using Ts::operator()...; };
};

int main() {
    std::println("Hello, world");

    int  a =  3;
    int b  = 18;
    std::println("A+B is {:d}", a+b);

    double pi = 3.141592;

    std::vector<double> jesus;
    jesus.push_back(pi);
    std::println("The last  element is {:f}", jesus.back());

    using Node = std::variant<Leaf, Decision>;

    Node mynode = Decision("indic" );

    std::visit(EnumResolve::overloaded{
        [](Leaf& leaf) {
            std::println("We have a leaf");
        },
        [](Decision& decision) {
            std::println("It is a decision, with indicator {:s}", decision.indicator);
        }
    }, mynode);
    
    return 0;
}
