# Tensor Library - C++

Biblioteca de tensores en C++ con soporte para operaciones matemáticas y transformaciones, similar a NumPy/PyTorch.

## Requisitos

- Compilador C++ compatible con C++11 o superior (g++, clang++, MSVC)
- CMake 3.10 o superior (opcional)
- CLion 2025.2.1 (recomendado) o cualquier IDE compatible con C++

## Estructura del Proyecto

```
TAREA 01/
├── tensor.h          # Declaración de la clase Tensor y transformaciones
├── tensor.cpp        # Implementación de los métodos
├── main.cpp          # Archivo principal con tests
├── CMakeLists.txt    # Configuración de CMake
└── README.md         # Este archivo
```

## Compilación y Ejecución

### Usando CLion (Recomendado)

1. Abre el proyecto en CLion
2. El IDE detectará automáticamente `CMakeLists.txt`
3. Presiona `Shift + F10` o haz clic en el botón de compilar y ejecutar
4. El ejecutable se generará en `cmake-build-debug/TAREA_01.exe`

## Tests Disponibles

El archivo `main.cpp` incluye 16 tests que puedes activar descomentando las líneas correspondientes en la función `main()`:

| Test | Descripción |
|------|-------------|
| `test_01()` | Tensor aleatorio 3D (2×3×4) |
| `test_02()` | Creación de tensor 2D con valores específicos |
| `test_03()` | Tensor de ceros 2D |
| `test_04()` | Tensor de unos 3D |
| `test_05()` | Tensor con rango de valores (arange) |
| `test_06()` | Suma de tensores 2D |
| `test_07()` | Broadcasting - suma (n×m) + (1×m) |
| `test_08()` | Broadcasting - resta (n×m) - (1×m) |
| `test_09()` | Multiplicación elemento a elemento |
| `test_10()` | Reshape con `view` |
| `test_11()` | Expansión de dimensiones con `unsqueeze` |
| `test_12()` | Concatenación de tensores |
| `test_13()` | Producto punto (dot) de vectores |
| `test_14()` | Multiplicación matricial (matmul) |
| `test_15()` | Aplicación de transformaciones (ReLU y Sigmoid) |
| `test_final()` | Pipeline completo de operaciones |

## Funcionalidades Principales

### Creación de Tensores

```cpp
// Tensor de ceros
Tensor A = Tensor::zeros({2, 3});
// [ 0 0 0 ]
// [ 0 0 0 ]

// Tensor de unos
Tensor B = Tensor::ones({2, 3, 4});

// Tensor aleatorio (valores entre min y max)
Tensor C = Tensor::random({2, 3}, 0.0, 1.0);

// Tensor con rango de valores
Tensor D = Tensor::arange(-5, 5);
// [ -5 -4 -3 -2 -1 0 1 2 3 4 ]

// Tensor con valores específicos
Tensor E = Tensor({2, 3}, {1, 2, 3, 4, 5, 6});
// [ 1 2 3 ]
// [ 4 5 6 ]
```

### Operaciones Aritméticas

```cpp
Tensor A = Tensor::random({2, 3}, 0, 1);
Tensor B = Tensor::random({2, 3}, 0, 1);

// Suma elemento a elemento
Tensor C = A + B;

// Resta elemento a elemento
Tensor D = A - B;

// Multiplicación elemento a elemento
Tensor E = A * B;

// Multiplicación por escalar
Tensor F = A * 2.5;
```

### Broadcasting

Operaciones entre tensores de diferentes tamaños:

```cpp
Tensor A = Tensor({10, 2}, {...});  // 10×2
Tensor B = Tensor({1, 2}, {-1, -2}); // 1×2

Tensor C = A + B;  // Broadcasting: B se expande a 10×2
```

### Operaciones Matriciales

```cpp
// Producto punto (vectores 1D)
Tensor A = Tensor::arange(1, 5);  // [1, 2, 3, 4]
Tensor B = Tensor::arange(2, 6);  // [2, 3, 4, 5]
Tensor C = dot(A, B);  // [40]

// Multiplicación matricial (tensores 2D)
Tensor A = Tensor({2, 3}, {1, 2, 3, 4, 5, 6});
Tensor B = Tensor({3, 2}, {1, 2, 3, 4, 5, 6});
Tensor C = matmul(A, B);  // Resultado: 2×2
```

### Manipulación de Forma

```cpp
// View (reshape sin copiar datos)
Tensor A = Tensor::arange(0, 12);  // [0, 1, 2, ..., 11]
Tensor B = A.view({3, 4});
// [ 0  1  2  3  ]
// [ 4  5  6  7  ]
// [ 8  9  10 11 ]

// Unsqueeze (agregar dimensión)
Tensor A = Tensor::arange(0, 3);  // Shape: {3}
Tensor B = A.unsqueeze(0);        // Shape: {1, 3}
Tensor C = A.unsqueeze(1);        // Shape: {3, 1}

// Concatenación
Tensor A = Tensor::ones({2, 3});
Tensor B = Tensor::zeros({2, 3});
Tensor C = Tensor::concat({A, B}, 1);  // Concatenar por columnas
// [ 1 1 1 0 0 0 ]
// [ 1 1 1 0 0 0 ]
```

### Transformaciones (Apply)

```cpp
Tensor A = Tensor::arange(-5, 5).view({2, 5});

// Aplicar ReLU
ReLU relu;
Tensor B = A.apply(relu);

// Aplicar Sigmoid
Sigmoid sigmoid;
Tensor C = B.apply(sigmoid);
```

#### Transformaciones Disponibles

- **ReLU**: `f(x) = max(0, x)`
- **Sigmoid**: `f(x) = 1 / (1 + e^(-x))`

### Pipeline Completo (test_final)

```cpp
// 1. Crear tensor de entrada 1000×20×20
Tensor A = Tensor::random({1000, 20, 20}, 0, 10);

// 2. Reshape a 1000×400
Tensor B = A.view({1000, 400});

// 3. Multiplicar por matriz 400×100
Tensor C = Tensor::random({400, 100}, 0, 10);
Tensor D = matmul(B, C);

// 4. Sumar bias 1×100
Tensor E = Tensor::random({1, 100}, 0, 10);
Tensor F = D + E;

// 5. Aplicar ReLU
ReLU relu;
Tensor G = F.apply(relu);

// 6. Multiplicar por matriz 100×10
Tensor H = Tensor::random({100, 10}, 0, 10);
Tensor I = matmul(G, H);

// 7. Sumar bias 1×10
Tensor J = Tensor::random({1, 10}, 0, 10);
Tensor K = I + J;

// 8. Aplicar Sigmoid
Sigmoid sigmoid;
Tensor output = K.apply(sigmoid);
```

## Notas Importantes

### Limitaciones

- **Dimensiones soportadas**: Solo 1D, 2D y 3D
- **Tamaño máximo**: Limitado por memoria disponible
- **Tipos de datos**: Solo `double` (64 bits)

### Gestión de Memoria

La clase `Tensor` implementa:
- **Constructor de copia**: Copia profunda de datos
- **Constructor de movimiento**: Transferencia eficiente de recursos
- **Destructor**: Liberación automática de memoria
- **Flag `owns_data`**: Para distinguir entre datos propios y vistas

### Uso de `view`

**IMPORTANTE**: `view` crea una vista **sin copiar datos** (`owns_data = false`). Debes asegurarte de que el tensor original permanezca vivo mientras uses la vista:

```cpp
// INCORRECTO - Tensor temporal destruido
Tensor A = Tensor::arange(-5, 5).view({2, 5});

// CORRECTO - Tensor original permanece vivo
Tensor temp = Tensor::arange(-5, 5);
Tensor A = temp.view({2, 5});
```

### Broadcasting

El broadcasting solo está implementado para tensores 2D con las siguientes reglas:
- `(n×m) op (1×m)` → resultado `(n×m)`
- `(1×m) op (n×m)` → resultado `(n×m)`



## Formato de Salida

Los tensores se imprimen automáticamente con `std::cout`:

```cpp
// 1D
Tensor A = Tensor::arange(0, 5);
cout << A;
// [ 0 1 2 3 4 ]

// 2D
Tensor B = Tensor::ones({2, 3});
cout << B;
// [ 1 1 1 ]
// [ 1 1 1 ]

// 3D
Tensor C = Tensor::zeros({2, 2, 3});
cout << C;
// Slice 0:
// [ 0 0 0 ]
// [ 0 0 0 ]
//
// Slice 1:
// [ 0 0 0 ]
// [ 0 0 0 ]
```

## Autor : gborjasb

Proyecto desarrollado para el curso de Programación 3 - TAREA 01

