#include "tensor.h"

void test_01 () {
    auto t = Tensor::random({2,3,4}, 1,10);

    cout << "Test 01: \n";
    cout << t;
    cout << "\n";
}

void test_02 () {
    auto t = Tensor({2,3},{1,2,34,56.2,3,0});

    cout << "Test 02: \n";
    cout << t;
    cout << "\n";
}

void test_03 () {
    auto t = Tensor::zeros({2,3});

    cout << "Test 03: \n";
    cout << t;
    cout << "\n";
}

void test_04 () {
    auto t = Tensor::ones({2,3,3});

    cout << "Test 04: \n";
    cout << t;
    cout << "\n";
}

void test_05 () {
    auto t = Tensor::arange(-5.1,5.1);

    cout << "Test 05: \n";
    cout << t;
    cout << "\n";
}

void test_06 () {
    auto a = Tensor::random({2,3}, 0,1);
    auto b = Tensor::random({2,3}, 0,1);

    cout << "Test 06: \n";
    cout << a << "\n";
    cout << b << "\n";
    cout << a + b << "\n";
    cout << "\n";
}

void test_07 () {
    auto a = Tensor({10,2}, {1,2,3,4,5,6,7,7,7,7,7,7,7,7,7,7,12,12,3,-1});
    auto b = Tensor({1,2}, {-1,-2});

    cout << "Test 07: \n";
    cout << a << "\n";
    cout << b << "\n";
    cout << a + b << "\n";
    cout << "\n";
}

void test_08 () {
    auto a = Tensor({10,2}, {1,2,3,4,5,6,7,7,7,7,7,7,7,7,7,7,12,12,3,-1});
    auto b = Tensor({1,2}, {10,5});

    cout << "Test 07: \n";
    cout << a << "\n";
    cout << b << "\n";
    cout << a - b << "\n";
    cout << "\n";
}

void test_09 () {
    auto a = Tensor({10,2}, {1,2,3,4,5,6,7,7,7,7,7,7,7,7,7,7,12,12,3,-1});
    auto b = Tensor({1,2}, {1, 3});

    cout << "Test 07: \n";
    cout << a << "\n";
    cout << b << "\n";
    cout << a * b << "\n";
    cout << "\n";
}

void test_10 () {
    Tensor A = Tensor::arange(0, 12);
    Tensor B = A.view({3, 4});

    cout << A << "\n\n";
    cout << B << "\n";
}

void test_11 () {
    Tensor A = Tensor::arange(0, 3);
    Tensor B = A.unsqueeze(0); // shape: {1,3}
    Tensor C = A.unsqueeze(1); // shape: {3,1}

    cout << A << "\n\n";
    cout << B << "\n\n";
    cout << C << "\n\n";
}

void test_12 () {
    Tensor A = Tensor::ones({2, 3});
    Tensor B = Tensor::zeros({2, 3});
    Tensor C = Tensor::concat({A, B}, 1);

    cout << A << "\n\n";
    cout << B << "\n\n";
    cout << C << "\n\n";
}

void test_13 () {
    Tensor A = Tensor::arange(1, 5);
    Tensor B = Tensor::arange(2, 6);
    Tensor C = dot(A,B);

    cout << A << "\n\n";
    cout << B << "\n\n";
    cout << C << "\n\n";
}

void test_14 () {
    Tensor A = Tensor({2,3},{1,2,3,4,5,6});
    Tensor B = Tensor({3,2},{1,2,3,4,5,6});
    Tensor C = matmul(A,B);

    cout << A << "\n\n";
    cout << B << "\n\n";
    cout << C << "\n\n";

}
void test_15 () {
    Tensor aux = Tensor::arange(-5, 5);
    Tensor A = aux.view({2,5});
    ReLU relu;
    Sigmoid sigmoid;
    Tensor B = A.apply(relu);
    Tensor C = B.apply(sigmoid);

    cout << A << "\n\n";
    cout << B << "\n\n";
    cout << C << "\n\n";
}
void test_final () {
    // 1. Crear un tensor de entrada de dimensiones 1000 × 20 ×20.
    Tensor A = Tensor::random({1000,20,20}, 0,10);

    // 2. Transformarlo a 1000 × 400 usando view.
    Tensor B = A.view({1000,400});

    // 3. Multiplicarlo por una matriz 400 × 100.
    Tensor C = Tensor::random({400,100}, 0,10);
    Tensor D = matmul(B, C);

    // 4. Sumar una matriz 1×100.
    Tensor E = Tensor::random({1,100}, 0,10);
    Tensor F = D + E;

    // 5. Aplicar la funcion ReLU.
    ReLU relu;
    Tensor G = F.apply(relu);

    // 6. Multiplicar por una matriz 100 × 10.
    Tensor H = Tensor::random({100, 10}, 0, 10);
    Tensor I = matmul(G, H);

    // 7. Sumar una matriz 1×10.
    Tensor J = Tensor::random({1, 10}, 0, 10);
    Tensor K = I + J;

    // 8. Aplicar la funcion Sigmoid.
    Sigmoid sigmoid;
    Tensor output = K.apply(sigmoid);

    cout << output;
}


int main() {
    // test_01();
    // test_02();
    // test_03();
    // test_04();
    // test_05();
    // test_06();
    // test_07();
    // test_08();
    // test_09();
    // test_10();
    // test_11();
    // test_12();
    // test_13();
    // test_14();
    // test_15();
    test_final();
    return 0;
}
