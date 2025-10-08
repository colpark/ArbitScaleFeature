# MAMBA-GINR Architecture Diagrams

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MAMBA-GINR Pipeline                                │
└─────────────────────────────────────────────────────────────────────────────┘

Input Image (H × W × C)
         │
         ▼
┌─────────────────────┐
│   Tokenization      │  Patch-based tokenization
│   (e.g., 4×4 patches)│  H×W → (H/4)×(W/4) patches
└─────────────────────┘
         │
         │  Tokens: (B, N, D)
         │  where N = num_patches
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ★ INNOVATION: Implicit Sequential Bias ★                  │
│                                                                               │
│   Input Tokens:   [T1] [T2] [T3] [T4] [T5] [T6] [T7] [T8] ...              │
│                     +    +    +    +    +    +    +    +                    │
│   Learnable LPs:  [L1]      [L2]      [L3]      [L4] ...                   │
│                     ║    ║    ║    ║    ║    ║    ║    ║                    │
│                     ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼                    │
│   Interleaved:   [L1] [T1] [T2] [L2] [T3] [T4] [L3] [T5] [T6] [L4] ...    │
│                                                                               │
│   Shape: (B, N+M, D) where M = num_lp                                       │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Bidirectional MAMBA Encoder                             │
│                                                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  Block 1: BiMamba (Forward + Reverse) → MLP → LayerNorm             │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                               ↓                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  Block 2: BiMamba (Forward + Reverse) → MLP → LayerNorm             │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                               ↓                                              │
│                              ...                                             │
│                               ↓                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  Block D: BiMamba (Forward + Reverse) → MLP → LayerNorm             │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│   Output: Encoded sequence (B, N+M, D)                                      │
│   Complexity: O(L) where L = N+M  ⚡ LINEAR COMPLEXITY!                     │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              ★ INNOVATION: Position-Aware Feature Extraction ★               │
│                                                                               │
│   Encoded:    [E_L1] [E_T1] [E_T2] [E_L2] [E_T3] [E_T4] [E_L3] ...        │
│                  ║                    ║                    ║                 │
│   Extract LPs:   ╚═══════════════════╩═══════════════════╬═══...           │
│                  ▼                    ▼                    ▼                 │
│   LP Features: [E_L1]              [E_L2]              [E_L3] ...          │
│                                                                               │
│   Shape: (B, M, D) - Compact position-aware representation!                 │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   Continuous Decoder (Hyponet)                               │
│                                                                               │
│   LP Features (B, M, D) ────┐                                               │
│                               │                                              │
│   Query Coordinates (B, Q, k)─┼──→ Cross-Attention ──→ MLP Decoder         │
│   (Arbitrary resolution!)     │                                              │
│                               │                                              │
│   Output: (B, Q, output_dim)  ← Can be ANY resolution!                     │
│                                                                               │
│   Examples:                                                                  │
│   - Train on 32×32  → Generate 64×64  (super-resolution)                   │
│   - Train on 32×32  → Generate 128×128 (4× super-resolution)               │
│   - Train on 32×32  → Generate ANY continuous query points                 │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
    Final Output
```

---

## Component Details

### 1. Learnable Position Token (LP) Mechanism

```
┌──────────────────────────────────────────────────────────────────┐
│  Learnable Position Tokens (Parameters)                          │
│                                                                   │
│  self.lps = nn.Parameter(torch.randn(num_lp, dim))              │
│                                                                   │
│  Shape: (M, D)                                                   │
│  M = number of LPs (typically 64-128)                           │
│  D = model dimension (typically 256-512)                        │
│                                                                   │
│  Initialized randomly, learned through backpropagation ✓        │
└──────────────────────────────────────────────────────────────────┘

Placement Strategies:

1. EQUIDISTANT (Most Common)
   Input:  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _  (20 tokens)
   LPs:    L   L   L   L   L                         (5 LPs)
   Result: L _ _ _ L _ _ _ L _ _ _ L _ _ _ L _ _ _

2. MIDDLE
   Input:  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
   LPs:            L L L L L                         (5 LPs in middle)
   Result: _ _ _ _ _ L L L L L _ _ _ _ _ _ _ _ _ _

3. N-GROUP (n=2)
   Input:  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
   LPs:    L L     L L     L L                       (3 groups of 2)
   Result: L L _ _ _ L L _ _ _ L L _ _ _ _ _ _ _ _
```

### 2. Bidirectional Mamba Block

```
Input Sequence x (B, L, D)
     │
     ├─────────────────────────────────┬───────────────────────────────┐
     │                                  │                                │
     ▼                                  ▼                                │
┌──────────┐                     ┌──────────┐                          │
│ Forward  │                     │ Reverse  │                          │
│  Mamba   │                     │  Mamba   │                          │
│          │                     │          │                          │
│  →→→→→   │                     │  ←←←←←   │                          │
└──────────┘                     └──────────┘                          │
     │                                  │                                │
     │     x_forward (B, L, D)         │  x_reverse (B, L, D)          │
     │                                  │                                │
     └────────────┬─────────────────────┘                                │
                  ▼                                                      │
           Average outputs                                              │
           out = (x_f + x_r) / 2                                        │
                  │                                                      │
                  ▼                                                      │
            (B, L, D)                                                    │
                  │                                                      │
                  ▼                                                      │
            ┌─────────┐                                                 │
            │   MLP   │                                                 │
            │  + GELU │                                                 │
            └─────────┘                                                 │
                  │                                                      │
                  ├──────────────────────────────────────────────────────┘
                  ▼
           LayerNorm + Residual
                  │
                  ▼
         Output (B, L, D)

Key Properties:
- Bidirectional context (unlike standard autoregressive)
- Linear complexity O(L) (unlike attention O(L²))
- Efficient state space model computation
```

### 3. LP Extraction Process

```
After Encoding:

Sequence: [E_L1, E_T1, E_T2, E_T3, E_L2, E_T4, E_T5, E_T6, E_L3, E_T7, ...]
Indices:  [  0,    1,    2,    3,    4,    5,    6,    7,    8,    9, ...]

LP Indices: [0, 4, 8, ...] (stored during initialization)

Extraction:
    lp_features = encoded_sequence[:, lp_indices, :]

Result: [E_L1, E_L2, E_L3, ...]  (B, M, D)

These features have:
✓ Interacted with all input tokens
✓ Absorbed positional information
✓ Learned task-specific position encodings
✓ Compact representation of sequence context
```

### 4. Continuous Decoder (Hyponet)

```
┌────────────────────────────────────────────────────────────────┐
│                    Continuous Decoder                           │
│                                                                  │
│  Input 1: LP Features (B, M, D)                                │
│  Input 2: Query Coordinates (B, Q, k)                          │
│                                                                  │
│  ┌──────────────────────────────────────────────────────┐     │
│  │  Cross-Attention                                      │     │
│  │  Query: From coordinates                              │     │
│  │  Key/Value: From LP features                         │     │
│  │                                                        │     │
│  │  Attention weights show which LPs are relevant       │     │
│  │  for each query position                             │     │
│  └──────────────────────────────────────────────────────┘     │
│                         ↓                                       │
│  ┌──────────────────────────────────────────────────────┐     │
│  │  MLP Decoder                                          │     │
│  │  Fourier features of coordinates + LP context        │     │
│  │  → Hidden layers                                     │     │
│  │  → Output                                            │     │
│  └──────────────────────────────────────────────────────┘     │
│                         ↓                                       │
│  Output: (B, Q, output_dim)                                   │
│                                                                  │
│  Q can be ANYTHING:                                            │
│  - 64×64 = 4,096 points                                        │
│  - 128×128 = 16,384 points                                     │
│  - 256×256 = 65,536 points                                     │
│  - Arbitrary irregular sampling                                │
└────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Example

Let's trace through a concrete example:

### Input: 32×32 RGB Image

```
Step 0: Input
├─ Image: (32, 32, 3)
└─ Goal: Generate 64×64 super-resolution

Step 1: Tokenization (4×4 patches)
├─ Patches: 32/4 × 32/4 = 8 × 8 = 64 patches
├─ Each patch: 4×4×3 = 48 values → project to D=256
└─ Tokens: (1, 64, 256)

Step 2: Add Learnable Position Tokens
├─ num_lp = 16 (using equidistant)
├─ Total sequence: 64 + 16 = 80 tokens
├─ Interleaved: [L, T, T, T, T, L, T, T, T, T, L, ...]
└─ Shape: (1, 80, 256)

Step 3: Bidirectional Mamba Encoding (6 layers)
├─ Each layer processes 80 tokens
├─ Complexity: O(80) per layer → Total O(480)
├─ Compare to Transformer: O(80²) = O(6400) per layer → Total O(38,400)
│  → Mamba is ~80× more efficient!
└─ Output: (1, 80, 256)

Step 4: Extract LP Features
├─ Extract 16 LP positions from 80 tokens
├─ LP indices: [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
└─ LP Features: (1, 16, 256)

Step 5: Query 64×64 Resolution
├─ Create coordinate grid: 64×64 = 4,096 points
├─ Coordinates: (1, 4096, 2) with values in [0, 1]
├─ Decoder uses LP features to generate output
└─ Output: (1, 4096, 3) → Reshape to (1, 64, 64, 3)

Result: 64×64 RGB image (2× super-resolution)! ✓
```

---

## Complexity Analysis

### Memory Complexity

```
┌────────────────────────────────────────────────────────────┐
│ Component          │ Shape              │ Memory (approx) │
├────────────────────┼────────────────────┼─────────────────┤
│ Input Tokens       │ (B, N, D)         │ B×N×D           │
│ Learnable LPs      │ (M, D)            │ M×D             │
│ Interleaved        │ (B, N+M, D)       │ B×(N+M)×D       │
│ Mamba States       │ (B, N+M, State)   │ B×(N+M)×State   │
│ LP Features        │ (B, M, D)         │ B×M×D           │
│ Query Coords       │ (B, Q, k)         │ B×Q×k           │
│ Output             │ (B, Q, out_dim)   │ B×Q×out_dim     │
└────────────────────────────────────────────────────────────┘

Total: O(B × max(N+M, Q) × D)

Example (B=4, N=256, M=64, D=512, Q=4096):
- Interleaved: 4 × 320 × 512 = 655,360 floats ≈ 2.5 MB
- LP Features: 4 × 64 × 512 = 131,072 floats ≈ 0.5 MB
- Very efficient! ✓
```

### Computational Complexity

```
┌────────────────────────────────────────────────────────────────┐
│ Operation                │ Complexity          │ Notes         │
├──────────────────────────┼─────────────────────┼───────────────┤
│ Tokenization             │ O(HWC)              │ Linear in img │
│ LP Interleaving          │ O(N+M)              │ Permutation   │
│ Mamba Encoding           │ O(D × (N+M) × D)    │ O(L) per dim  │
│   Per layer              │ O((N+M) × D)        │ Linear!       │
│   Total (depth layers)   │ O(depth × (N+M) × D)│               │
│ LP Extraction            │ O(M)                │ Indexing      │
│ Decoding                 │ O(Q × M × D)        │ Cross-attn    │
└────────────────────────────────────────────────────────────────┘

Overall: O(depth × (N+M) × D + Q × M × D)

Comparison to Transformer:
- Transformer Encoding: O(depth × (N+M)² × D)
- Mamba Encoding: O(depth × (N+M) × D)
- Speedup: O((N+M)) → For 320 tokens, ~320× faster encoding! ⚡
```

---

## Training Dynamics

### What Gets Learned?

```
┌──────────────────────────────────────────────────────────────┐
│ 1. Learnable Position Tokens (LPs)                          │
│    - Learn to encode optimal positional representations      │
│    - Adapt to task-specific spatial patterns                │
│    - Develop diversity to cover different positions         │
│                                                              │
│ 2. Mamba Encoder                                            │
│    - Learn to process sequences bidirectionally             │
│    - Develop efficient state space transitions              │
│    - Encode both content and position information           │
│                                                              │
│ 3. Decoder (Hyponet)                                        │
│    - Learn to interpret LP features                         │
│    - Map coordinates to outputs using LP context            │
│    - Generalize to unseen resolutions                       │
└──────────────────────────────────────────────────────────────┘

Gradient Flow:

Loss ← Output ← Decoder ← LP Features ← Mamba ← Interleaved ← LPs
  ↓                ↓                        ↓                    ↓
  └────────────────┴────────────────────────┴────────────────────┘
              All components updated via backprop!

Key: LPs receive gradients that teach them useful positions
```

---

## Visualization of LP Learning

```
Initialization (Random):
LP1: [0.23, -0.45, 0.12, ...]  Random noise
LP2: [0.67, 0.34, -0.89, ...]  Random noise
LP3: [-0.23, 0.56, 0.45, ...]  Random noise
...
Similarity: High (random vectors often similar)

After Training (Learned):
LP1: [0.92, 0.12, -0.03, ...]  Specialized representation
LP2: [-0.23, 0.89, 0.45, ...]  Different from LP1
LP3: [0.45, -0.67, 0.78, ...]  Different from both
...
Similarity: Low (diverse, specialized)

Each LP learns to represent a different aspect of position!
```

---

## Key Architectural Decisions

### 1. Why Bidirectional Mamba?

```
Unidirectional (→):
- Only sees past context
- Position i knows about positions 1...i-1
- Limited for images (no natural ordering)

Bidirectional (→ + ←):
- Sees both past and future
- Position i knows about all positions
- Better for images and volumes ✓
```

### 2. Why Interleave Instead of Concatenate?

```
Concatenation:
[All Input Tokens] [All LP Tokens]
- LPs only see inputs in one direction
- Limited interaction

Interleaving:
[LP] [Input] [Input] [LP] [Input] [Input] [LP]
- LPs distributed throughout sequence
- Rich bidirectional interaction ✓
- Each LP has local and global context
```

### 3. Why Extract LPs Instead of Using All Tokens?

```
Use All Tokens:
- 320 tokens for decoder
- Lots of redundant information
- Slower decoding
- Harder to generalize

Use Only LPs:
- 64 tokens for decoder ✓
- Compact, informative representation ✓
- Faster decoding ✓
- Better generalization ✓
- LPs learned to summarize position info
```

---

This architecture enables MAMBA-GINR to achieve:
✓ Efficient encoding (linear complexity)
✓ Learnable positional bias
✓ Arbitrary-scale generation
✓ State-of-the-art performance
