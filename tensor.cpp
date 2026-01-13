//
// Created by gmbor on 12/01/2026.
//

#include "tensor.h"

Tensor::Tensor() {
    shape = nullptr;
    dims = 0;
    data = nullptr;
    owns_data = false;
}

// Constructores
Tensor::Tensor(const std::vector<size_t> &shape, const std::vector<double> &values) {
    // Dimension
    if (shape.size() > 3 || shape.size() == 0) {
        throw std::invalid_argument("Shape size must be between 1 and 3 \n");
    }

    dims = shape.size();


    // Tamaño y valores
    size_t multiplier = 1;
    for (int i = 0; i < shape.size(); i++) multiplier *= shape[i];
    if (multiplier != values.size()) {
        throw std::invalid_argument("Shape product must be the number of values \n");
    }

    // Crear el tensor
    this->shape = new size_t[shape.size()];
    this->data = new double[multiplier];
    for (int i = 0; i < shape.size(); i++) this->shape[i] = shape[i];
    for (int i = 0; i < values.size(); i++) this->data[i] = values[i];
    owns_data = true;
}

Tensor::Tensor(const Tensor& other) {
    dims = other.dims;
    owns_data = true;

    if (dims == 0) {
        shape = nullptr;
        data = nullptr;
        return;
    }

    shape = new size_t[dims];
    for (size_t i = 0; i < dims; ++i) shape[i] = other.shape[i];

    size_t n = other.shape_product();
    data = new double[n];
    for (size_t i = 0; i < n; ++i) data[i] = other.data[i];
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) return *this;

    // liberar lo actual
    delete[] shape;
    if (owns_data) delete[] data;

    dims = other.dims;
    owns_data = true;

    if (dims == 0) {
        shape = nullptr;
        data = nullptr;
        return *this;
    }

    shape = new size_t[dims];
    for (size_t i = 0; i < dims; ++i) shape[i] = other.shape[i];

    size_t n = other.shape_product();
    data = new double[n];
    for (size_t i = 0; i < n; ++i) data[i] = other.data[i];

    return *this;
}

Tensor::Tensor(Tensor&& other) noexcept {
    shape = other.shape;
    dims = other.dims;
    data = other.data;
    owns_data = other.owns_data;

    other.shape = nullptr;
    other.data = nullptr;
    other.dims = 0;
    other.owns_data = false;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this == &other) return *this;

    delete[] shape;
    if (owns_data) delete[] data;

    shape = other.shape;
    dims = other.dims;
    data = other.data;
    owns_data = other.owns_data;

    other.shape = nullptr;
    other.data = nullptr;
    other.dims = 0;
    other.owns_data = false;

    return *this;
}



Tensor::~Tensor() {
    delete[] shape;

    if (owns_data) delete[] data;
}

size_t Tensor::shape_product() const {
    size_t multiplier = 1;
    for (size_t i = 0; i < dims; i++) {
        multiplier *= this->shape[i];
    }
    return multiplier;
}

// Métodos estáticos

Tensor Tensor::zeros(const std::vector<size_t> &shape) {
    size_t multiplier = 1;
    for (size_t i = 0; i < shape.size(); i++) {
        multiplier *= shape[i];
    }

    return Tensor(shape, vector<double>(multiplier, 0));
}

Tensor Tensor::ones(const std::vector<size_t> &shape) {
    size_t multiplier = 1;
    for (size_t i = 0; i < shape.size(); i++) {
        multiplier *= shape[i];
    }

    return Tensor(shape, vector<double>(multiplier, 1));
}

Tensor Tensor::random(const std::vector<size_t> &shape, double min, double max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(min, max);

    size_t multiplier = 1;
    for (size_t i = 0; i < shape.size(); i++) {
        multiplier *= shape[i];
    }

    std::vector<double> values;
    values.reserve(multiplier);
    for (size_t i = 0; i < multiplier; i++) {
        values.emplace_back(dist(gen));
    }

    return Tensor(shape, values);
}

Tensor Tensor::arange(double min, double max) {
    vector<double> values;
    auto aux = min;
    while (aux < max) {
        values.emplace_back(aux);
        aux++;
    }

    return Tensor({values.size()}, values);
}

Tensor Tensor::operator+(const Tensor &other) const {
    if (this->dims != other.dims) {
        throw std::invalid_argument("Dimensions must be equal \n");
    }

    // Shape coincidence
    bool same_shape = true;
    for (size_t d = 0; d < dims; ++d)
        if (shape[d] != other.shape[d]) same_shape = false;

    if (same_shape) {
        vector<size_t> out_shape(dims);
        for (size_t d = 0; d < dims; ++d) out_shape[d] = shape[d];

        vector<double> values;

        for (size_t i = 0; i < shape_product(); ++i)
            values.emplace_back(data[i] + other.data[i]);

        return Tensor(out_shape, values);
    }

    // 2D case
    if (dims == 2) {
        size_t rA = shape[0], cA = shape[1];
        size_t rB = other.shape[0], cB = other.shape[1];

        vector<double> values;

        // (n x m) + (1 x m)
        if (rB == 1 && cA == cB) {
            for (size_t i = 0; i < rA; ++i)
                for (size_t j = 0; j < cA; ++j)
                    values.emplace_back(data[i * cA + j] + other.data[j]);
            return Tensor({rA, cA}, values);
        }

        // (1 x m) + (n x m)
        if (rA == 1 && cA == cB) {
            for (size_t i = 0; i < rB; ++i)
                for (size_t j = 0; j < cB; ++j)
                    values.push_back(data[j] + other.data[i * cB + j]);
            return Tensor({rB, cB}, values);
        }
    }

    throw std::invalid_argument("operator+: incompatible shapes");
}

Tensor Tensor::operator-(const Tensor &other) {
    if (this->dims != other.dims) {
        throw std::invalid_argument("Dimensions must be equal \n");
    }

    // Shape coincidence
    bool same_shape = true;
    for (size_t d = 0; d < dims; ++d)
        if (shape[d] != other.shape[d]) same_shape = false;

    if (same_shape) {
        vector<size_t> out_shape(dims);
        for (size_t d = 0; d < dims; ++d) out_shape[d] = shape[d];

        vector<double> values;

        for (size_t i = 0; i < shape_product(); ++i)
            values.push_back(data[i] - other.data[i]);

        return Tensor(out_shape, values);
    }

    // 2D case
    if (dims == 2) {
        size_t rA = shape[0], cA = shape[1];
        size_t rB = other.shape[0], cB = other.shape[1];

        vector<double> values;

        // (n x m) + (1 x m)
        if (rB == 1 && cA == cB) {
            for (size_t i = 0; i < rA; ++i)
                for (size_t j = 0; j < cA; ++j)
                    values.push_back(data[i * cA + j] - other.data[j]);
            return Tensor({rA, cA}, values);
        }

        // (1 x m) + (n x m)
        if (rA == 1 && cA == cB) {
            for (size_t i = 0; i < rB; ++i)
                for (size_t j = 0; j < cB; ++j)
                    values.push_back(data[j] - other.data[i * cB + j]);
            return Tensor({rB, cB}, values);
        }
    }

    throw std::invalid_argument("operator+: incompatible shapes");
}

Tensor Tensor::operator*(const Tensor &other) {
    if (this->dims != other.dims) {
        throw std::invalid_argument("Dimensions must be equal \n");
    }

    // Shape coincidence
    bool same_shape = true;
    for (size_t d = 0; d < dims; ++d)
        if (shape[d] != other.shape[d]) same_shape = false;

    if (same_shape) {
        vector<size_t> out_shape(dims);
        for (size_t d = 0; d < dims; ++d) out_shape[d] = shape[d];

        vector<double> values;

        for (size_t i = 0; i < shape_product(); ++i)
            values.push_back(data[i] * other.data[i]);

        return Tensor(out_shape, values);
    }

    // 2D case
    if (dims == 2) {
        size_t rA = shape[0], cA = shape[1];
        size_t rB = other.shape[0], cB = other.shape[1];

        vector<double> values;

        // (n x m) + (1 x m)
        if (rB == 1 && cA == cB) {
            for (size_t i = 0; i < rA; ++i)
                for (size_t j = 0; j < cA; ++j)
                    values.emplace_back(data[i * cA + j] * other.data[j]);
            return Tensor({rA, cA}, values);
        }

        // (1 x m) + (n x m)
        if (rA == 1 && cA == cB) {
            for (size_t i = 0; i < rB; ++i)
                for (size_t j = 0; j < cB; ++j)
                    values.emplace_back(data[j] * other.data[i * cB + j]);
            return Tensor({rB, cB}, values);
        }
    }

    throw std::invalid_argument("operator+: incompatible shapes");
}

Tensor Tensor::operator*(double value) {
    vector<size_t> shape;
    vector<double> values;

    for (size_t i = 0; i < this->dims; i++) {
        shape.emplace_back(this->shape[i]);
    }
    for (int i = 0; i < this->shape_product(); i++) {
        values.emplace_back(this->data[i] * value);
    }

    return Tensor(shape, values);
}

Tensor Tensor::view(const std::vector<size_t> &shape_) const {
    size_t product = 1;
    for (size_t d: shape_)
        product *= d;

    if (product != this->shape_product())
        throw std::invalid_argument("Product of shapes must coincide");

    size_t *s = new size_t[shape_.size()];
    for (int i = 0; i < shape_.size(); i++) {
        s[i] = shape_[i];
    }

    Tensor t;
    t.dims = shape_.size();
    t.shape = s;
    t.data = this->data;
    t.owns_data = false;

    return t;
}


Tensor Tensor::unsqueeze(size_t position) {
    if (dims == 3) {
        throw std::invalid_argument("The maximum number of dimensions is 3");
    }

    if (position > dims) {
        throw std::invalid_argument("Position out of range");
    }

    vector<size_t> s(dims + 1);

    for (size_t i = 0; i < dims + 1; i++) {
        if (i < position) { s[i] = shape[i]; }
        else if (i == position) { s[i] = 1; }
        else { s[i] = shape[i - 1]; }
    }

    return this->view(s);
}

    Tensor Tensor::concat(const vector<Tensor> &tensors, size_t axis) {
        if (tensors.empty())
            throw invalid_argument("empty list");

        const Tensor &base = tensors[0];

        if (base.dims == 0 || base.dims > 3)
            throw invalid_argument("dims invalid");
        if (axis >= base.dims)
            throw invalid_argument("axis out of range");

        // Validaciones
        for (const auto &t: tensors) {
            if (t.dims != base.dims)
                throw invalid_argument("dims mismatch");
            for (size_t d = 0; d < base.dims; ++d) {
                if (d == axis) continue;
                if (t.shape[d] != base.shape[d])
                    throw invalid_argument("incompatible shapes");
            }
        }

        // New shape
        vector<size_t> new_shape(base.dims);
        for (size_t d = 0; d < base.dims; ++d) new_shape[d] = base.shape[d];

        size_t sum_axis = 0;
        for (const auto &t: tensors) sum_axis += t.shape[axis];
        new_shape[axis] = sum_axis;

        // Total size
        size_t new_total = 1;
        for (size_t d: new_shape) new_total *= d;

        vector<double> values;

        auto id1 = [](const Tensor &t, size_t i) -> size_t {
            return i;
        };
        auto id2 = [](const Tensor &t, size_t i, size_t j) -> size_t {
            return i * t.shape[1] + j;
        };
        auto id3 = [](const Tensor &t, size_t i, size_t j, size_t k) -> size_t {
            return i * t.shape[1] * t.shape[2] + j * t.shape[2] + k;
        };

        // 1D
        if (base.dims == 1) {
            for (const auto &t: tensors)
                for (size_t i = 0; i < t.shape[0]; ++i)
                    values.emplace_back(t.data[id1(t, i)]);
            return Tensor(new_shape, values);
        }

        // 2D
        if (base.dims == 2) {
            if (axis == 0) {
                for (const auto &t: tensors)
                    for (size_t i = 0; i < t.shape[0]; ++i)
                        for (size_t j = 0; j < t.shape[1]; ++j)
                            values.emplace_back(t.data[id2(t, i, j)]);
            } else {
                // axis == 1
                size_t rows = base.shape[0];
                for (size_t i = 0; i < rows; ++i)
                    for (const auto &t: tensors)
                        for (size_t j = 0; j < t.shape[1]; ++j)
                            values.emplace_back(t.data[id2(t, i, j)]);
            }
            return Tensor(new_shape, values);
        }

        // 3D
        // axis == 0
        if (axis == 0) {
            for (const auto &t: tensors)
                for (size_t i = 0; i < t.shape[0]; ++i)
                    for (size_t j = 0; j < t.shape[1]; ++j)
                        for (size_t k = 0; k < t.shape[2]; ++k)
                            values.emplace_back(t.data[id3(t, i, j, k)]);
            return Tensor(new_shape, values);
        }

        // axis == 1
        if (axis == 1) {
            size_t X = base.shape[0];
            for (size_t i = 0; i < X; ++i)
                for (const auto &t: tensors)
                    for (size_t j = 0; j < t.shape[1]; ++j)
                        for (size_t k = 0; k < t.shape[2]; ++k)
                            values.emplace_back(t.data[id3(t, i, j, k)]);
            return Tensor(new_shape, values);
        }

        // axis == 2
        size_t X = base.shape[0];
        size_t Y = base.shape[1];
        for (size_t i = 0; i < X; ++i)
            for (size_t j = 0; j < Y; ++j)
                for (const auto &t: tensors)
                    for (size_t k = 0; k < t.shape[2]; ++k)
                        values.emplace_back(t.data[id3(t, i, j, k)]);

        return Tensor(new_shape, values);
    }


Tensor dot(const Tensor &a, const Tensor &b) {
    if (a.dims != 1 || b.dims != 1) {
        throw std::invalid_argument("dimensions must be equal to 1");
    }

    if (a.shape_product() != b.shape_product()) {
        throw std::invalid_argument("shapes must be equal");
    }

    double result = 0;
    for (int i = 0; i < a.shape_product(); i++) {
        result = result + a.data[i] * b.data[i];
    }

    // Escalar representado como tensor 1D de tamaño 1
    return Tensor({1}, {result});
}

Tensor matmul(const Tensor &a, const Tensor &b) {
    // Validaciones
    if (a.dims != 2 || b.dims != 2)
        throw std::invalid_argument("both tensors must be 2D");

    size_t N = a.shape[0];
    size_t N1 = a.shape[1];
    size_t N2 = b.shape[0];
    size_t M = b.shape[1];

    if (N1 != N2)
        throw std::invalid_argument("incompatible shapes");

    // Shape del resultado
    vector<size_t> out_shape = {N, M};
    vector<double> values(N * M, 0.0);

    // Multiplicación matricial clásica
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < M; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < N1; ++k) {
                sum += a.data[i * N1 + k] * b.data[k * M + j];
            }
            values[i * M + j] = sum;
        }
    }

    return Tensor(out_shape, values);
}


Tensor Tensor::apply(const TensorTransform &transform) const {
    vector<size_t> out_shape(dims);
    for (size_t i = 0; i < dims; ++i)
        out_shape[i] = shape[i];

    vector<double> values;

    for (size_t i = 0; i < shape_product(); ++i)
        values.emplace_back(transform.apply(data[i]));

    return Tensor(out_shape, values);
}


std::ostream& operator<<(std::ostream& os, const Tensor& t)
{
    //
    if (t.dims == 1) {
        os << "[ ";
        for (size_t i = 0; i < t.shape[0]; ++i)
            os << t.data[i] << " ";
        os << "]";
        return os;
    }

    //
    if (t.dims == 2) {
        size_t rows = t.shape[0];
        size_t cols = t.shape[1];

        for (size_t i = 0; i < rows; ++i) {
            os << "[ ";
            for (size_t j = 0; j < cols; ++j)
                os << t.data[i * cols + j] << " ";
            os << "]\n";
        }
        return os;
    }

    //
    if (t.dims == 3) {
        size_t A = t.shape[0];
        size_t B = t.shape[1];
        size_t C = t.shape[2];

        for (size_t i = 0; i < A; ++i) {
            os << "Slice " << i << ":\n";
            for (size_t j = 0; j < B; ++j) {
                os << "[ ";
                for (size_t k = 0; k < C; ++k)
                    os << t.data[i * B * C + j * C + k] << " ";
                os << "]\n";
            }
            os << "\n";
        }
        return os;
    }
    return os;
}
