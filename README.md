# Neural Fractal Anchoring (NFA) Engine

Sistema C++/CUDA para compresion extrema de pesos de redes neuronales. Expande ~200 MB de datos comprimidos a pesos completos en VRAM usando operaciones tensoriales en GPU.

## Arquitectura

El motor NFA combina tres tecnicas de compresion en un pipeline de tres etapas:

### 1. Fractal Anchoring (Anclaje Fractal)
En lugar de guardar los pesos directamente, se almacenan **semillas fractales** (transformaciones afines de un IFS - Sistema de Funciones Iteradas). Al cargar el modelo, la GPU ejecuta el "juego del caos" para reconstruir cada peso a partir de las semillas.

- Entrada: Matriz de pesos `[rows x cols]`
- Salida: `N` semillas fractales (~64 por capa)
- Compresion: >1000x por capa individual

### 2. Cuantizacion Topologica (Hilbert Curve Indexing)
Los residuos (error entre pesos originales y reconstruccion fractal) se codifican como **indices en una curva de Hilbert**. Grupos de pesos se mapean a un solo indice que captura centroide y varianza.

- Preserva localidad espacial de los pesos
- Codificacion por grupos para compresion adicional
- Decodificacion paralela en GPU

### 3. Engramas de ADN (Dynamic Mixture of Experts)
Los residuos de segundo orden se descomponen en **micro-expertos** con bases de bajo rango. Un diccionario de activacion permite reconstruir pesos **Just-In-Time** segun la entrada.

- Descomposicion SVD randomizada por bloques
- Recuperacion semantica por similitud coseno en GPU
- Reconstruccion via producto tensorial de bases

## Pipeline de Expansion

```
[Archivo NFA ~200MB]
        |
        v
  Stage 1: Fractal Expand (IFS chaos game en GPU)
        |
        v
  Stage 2: Topological Correction (Hilbert decode + sum)
        |
        v
  Stage 3: Engram Refinement (JIT MoE por input)
        |
        v
[Pesos completos en VRAM]
```

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Requisitos
- CUDA Toolkit >= 11.0
- CMake >= 3.18
- GPU con Compute Capability >= 7.0 (Volta+)

## Ejecutar Demo/Benchmark

```bash
./build/nfa_demo
```

Ejecuta benchmarks de cada subsistema y del pipeline completo con modelos sinteticos.

## Estructura del Proyecto

```
include/
  nfa_types.h              # Tipos fundamentales y configuracion
  fractal_anchor.h         # API de compresion/expansion fractal
  topological_quantizer.h  # API de cuantizacion topologica
  dna_engram.h             # API del sistema de engramas/MoE
  nfa_engine.h             # Motor unificado NFA

src/
  kernels/
    fractal_kernels.cu     # Kernels CUDA: IFS, producto tensorial
    hilbert_kernels.cu     # Kernels CUDA: curva de Hilbert encode/decode
    engram_kernels.cu      # Kernels CUDA: activacion de expertos, low-rank
  fractal_anchor.cu        # Implementacion host del sistema fractal
  topological_quantizer.cu # Implementacion host de cuantizacion topologica
  dna_engram.cu            # Implementacion host del sistema de engramas
  nfa_engine.cu            # Motor unificado: pipeline de 3 etapas

demo/
  main.cu                  # Benchmarks y demo
```

## API

```cpp
#include "nfa_engine.h"

// Configurar
nfa::NFAConfig config = nfa::default_config();
nfa::NFAEngine engine(config);

// Comprimir modelo (vector de capas: {puntero_pesos, {filas, columnas}})
engine.compress_model(layers);
engine.save("model.nfa");

// Cargar y expandir a VRAM
engine.load("model.nfa");
auto gpu_buffers = engine.expand_all_gpu();  // Pesos completos en GPU

// O expansion JIT por capa segun input
auto weights = engine.jit_forward_gpu(layer_id, d_input, input_dim);
```
