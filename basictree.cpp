#include <string>
#include <variant>
#include <memory>
#include <vector>
#include <print>

struct Decision;
struct Leaf;

using Node = std::variant<Decision, Leaf>;


struct Decision {
    size_t feature;
    double threshold;
    std::unique_ptr<Node> left;
    std::unique_ptr<Node> right;
};

struct Leaf {
    double residual;
};

namespace EnumEval {
    template<typename... Ts>
    struct overloaded: Ts... { using Ts::operator()...; };
}

double traverse(Node& head, std::vector<double>& features) {
    std::println("traversing");

    return std::visit(EnumEval::overloaded{
        [&features](Decision& decision) {
            std::println("It was a decision node");
            if (features.at(decision.feature) < decision.threshold) {
                std::println("We go down  the left line");
                return traverse(*decision.left, features);
            } else {
                std::println("We go down  the right line");
                return traverse(*decision.right, features);
            }
        },
        [](Leaf& leaf) {
            std::println("Yup it ");

            return leaf.residual;
        },
    }, head);
}

int main() {
    std::unique_ptr<Node> branch_0a_0a = std::make_unique<Node>(Leaf{60});
    std::unique_ptr<Node> branch_0a_0b = std::make_unique<Node>(Leaf{-60});


    std::unique_ptr<Node> branch_0a = std::make_unique<Node>(Decision{1, 0.5, std::move(branch_0a_0a), std::move(branch_0a_0b)});

    std::unique_ptr<Node> branch_0b = std::make_unique<Node>(Leaf{30});
    std::unique_ptr<Node> head = std::make_unique<Node>(Decision{0, 0.5, std::move(branch_0a), std::move(branch_0b)});

    std::vector<double> features = {0.1, 0.9, 0.9};
    std::println("Created structure, parsing");
    double result = traverse(*head, features);
    std::println("The result is {:g}", result);
}