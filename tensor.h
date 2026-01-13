//
// Created by gmbor on 12/01/2026.
//

#ifndef TAREA_01_TENSOR_H
#define TAREA_01_TENSOR_H

#include <iostream>
#include <stdexcept>
#include <vector>
#include <random>

using namespace std;

class TensorTransform {
public:
    virtual double apply(double x) const = 0;

    virtual ~TensorTransform() = default;
};

class Tensor {
private:
    size_t *shape;
    size_t dims;
    double *data;
    bool owns_data;

public:
    // Constructor por defecto: necesario para view
    Tensor();

    // Constructores
    Tensor(const std::vector<size_t> &shape,
           const std::vector<double> &values);

    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);

    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;


    // Destructor
    ~Tensor();

    // Shape product
    size_t shape_product() const;

    // Métodos estáticos
    static Tensor zeros(const std::vector<size_t> &shape);

    static Tensor ones(const std::vector<size_t> &shape);

    static Tensor random(const std::vector<size_t> &shape, double min, double max);

    static Tensor arange(double min, double max);

    // Sobrecarga de operadores
    Tensor operator+(const Tensor &other) const;

    Tensor operator-(const Tensor &other);

    Tensor operator*(const Tensor &other);

    Tensor operator*(double value);


    // View y Unsqueeze
    Tensor view(const std::vector<size_t> &shape_) const;

    Tensor unsqueeze(size_t position);

    // Concatenar
    static Tensor concat(const vector<Tensor> &tensors, size_t position);

    // Funciones amigas
    friend Tensor dot(const Tensor &a, const Tensor &b);

    friend Tensor matmul(const Tensor &a, const Tensor &b);

    // Apply
    Tensor apply(const TensorTransform& transform) const;

    // Impresion
    friend std::ostream& operator<<(std::ostream&os, const Tensor &t);
};


class ReLU : public TensorTransform {
public:
    double apply(double x) const override {
        return x > 0 ? x : 0;
    }
};

class Sigmoid : public TensorTransform {
public:
    double apply(double x) const override {
        return 1.0 / (1.0 + std::exp(-x));
    }
};


#endif //TAREA_01_TENSOR_H
