# Matrices & Operators

## Linear Operator Interface
For the use with this packages, operators are not restricted to be actual Matrices. This is due to the effect that for many operators it is impractical to store the full matrix. Examples for this are Fourier transforms or Wavelet transforms which frequently arise in areas such as Image Processing and MRI.

In order to work with the implemented solvers, the following methods need to be implemented for an operator `A`:
* `*(A,x)`
* `adjoint(A)`
Moreover, the sparsifying transformations also recquire the implementation of the function `\(A,x)`.
For an easy way to generate matrix-free operators, have a look at packages such as [LinearOperators.jl](https://github.com/JuliaSmoothOptimizers/LinearOperators.jl) and
[LinearMaps.jl](https://github.com/Jutho/LinearMaps.jl).

## Implemented Operators
For convenience, some common operators are implemented in this package. These include
* DCT-II and DCT-IV
* DST
* FFT
* Wavelet transform

To build these operators, one can use the method `linearOperator(op::AbstractString, shape)` - e.g.
```julia
  shape = (256,256)
  op = linearOperator("FFT", shape)
```
Here `shape` is the size of the Array to be transformed. Valid operate names are "FFT", "DCT-II", "DCT-IV", "DST" and "Wavelet".
Alternatively, one can directly call the constructors for the desired operator.
