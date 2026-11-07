"""저장 가중치와 F32 상태를 사용하는 decoder 및 출력 회전 연산."""
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def take_row(Matrix, Output, row, WIDTH: tl.constexpr, TILE: tl.constexpr):
    offsets = tl.program_id(0) * TILE + tl.arange(0, TILE)
    tl.store(Output + offsets, tl.load(Matrix + row * WIDTH + offsets,
                                      offsets < WIDTH, 0).to(tl.float32), offsets < WIDTH)


@triton.jit
def matvec(Matrix, Vector, Bias, Output, ROWS: tl.constexpr, COLS: tl.constexpr,
           BIAS: tl.constexpr, TILE_R: tl.constexpr, TILE_C: tl.constexpr):
    rows = tl.program_id(0) * TILE_R + tl.arange(0, TILE_R)
    lanes = tl.arange(0, TILE_C)
    sums = tl.full((TILE_R,), 0, tl.float32)
    for start in range(tl.cdiv(COLS, TILE_C)):
        cols = start * TILE_C + lanes
        weights = tl.load(Matrix + rows[:, None] * COLS + cols[None, :],
                          (rows[:, None] < ROWS) & (cols[None, :] < COLS), 0).to(tl.float32)
        values = tl.load(Vector + cols, cols < COLS, 0)
        sums += tl.sum(weights * values[None, :], axis=1)
    if BIAS:
        sums += tl.load(Bias + rows, rows < ROWS, 0)
    tl.store(Output + rows, sums, rows < ROWS)


@triton.jit
def normalization(Input, Gain, Bias, Output, WIDTH: tl.constexpr,
                  EPSILON: tl.constexpr, CENTER: tl.constexpr,
                  BIAS: tl.constexpr, TILE: tl.constexpr):
    group = tl.program_id(0)
    columns = tl.arange(0, TILE)
    valid = columns < WIDTH
    x = tl.load(Input + group * WIDTH + columns, valid, 0)
    if CENTER:
        x = tl.where(valid, x - tl.sum(x, 0) / WIDTH, 0)
    inverse = tl.rsqrt(tl.sum(x * x, 0) / WIDTH + EPSILON)
    out = x * inverse * tl.load(Gain + columns, valid, 0)
    if BIAS:
        out += tl.load(Bias + columns, valid, 0)
    tl.store(Output + group * WIDTH + columns, out, valid)


@triton.jit
def rotary_position(Input, Output, position, HEAD_DIM: tl.constexpr,
                    WIDTH: tl.constexpr, ROTARY: tl.constexpr,
                    BASE: tl.constexpr, INTERLEAVED: tl.constexpr, TILE: tl.constexpr):
    index = tl.program_id(0) * TILE + tl.arange(0, TILE)
    local = index % HEAD_DIM
    if INTERLEAVED:
        pair = local // 2
        other = tl.where(local % 2 == 0, local + 1, local - 1)
        sign = tl.where(local % 2 == 0, -1., 1.)
    else:
        pair = local % (ROTARY // 2)
        other = tl.where(local < ROTARY // 2, local + ROTARY // 2, local - ROTARY // 2)
        sign = tl.where(local < ROTARY // 2, -1., 1.)
    angle = position * tl.exp(-tl.log(BASE) * (2. * pair / ROTARY))
    original = tl.load(Input + index, index < WIDTH, 0)
    partner = tl.load(Input + index - local + other, (index < WIDTH) & (local < ROTARY), 0)
    rotated = original * tl.cos(angle) + sign * partner * tl.sin(angle)
    tl.store(Output + index, tl.where(local < ROTARY, rotated, original), index < WIDTH)


@triton.jit
def cache_write(Key, Value, Keys, Values, position, WIDTH: tl.constexpr, TILE: tl.constexpr):
    i = tl.program_id(0) * TILE + tl.arange(0, TILE)
    tl.store(Keys + position * WIDTH + i, tl.load(Key + i, i < WIDTH, 0), i < WIDTH)
    tl.store(Values + position * WIDTH + i, tl.load(Value + i, i < WIDTH, 0), i < WIDTH)


@triton.jit
def causal_attention(Query, Keys, Values, Output, position,
                     Q_HEADS: tl.constexpr, KV_HEADS: tl.constexpr,
                     DIM: tl.constexpr, SCALE: tl.constexpr,
                     TILE_T: tl.constexpr, TILE_D: tl.constexpr):
    head = tl.program_id(0)
    kv = head // (Q_HEADS // KV_HEADS)
    channels = tl.arange(0, TILE_D)
    q = tl.load(Query + head * DIM + channels, channels < DIM, 0)
    numerator = tl.full((TILE_D,), 0., tl.float32)
    denominator = 0.
    maximum = -float('inf')
    for tile in range(tl.cdiv(position + 1, TILE_T)):
        times = tile * TILE_T + tl.arange(0, TILE_T)
        offsets = times[:, None] * KV_HEADS * DIM + kv * DIM + channels[None, :]
        valid = (times[:, None] <= position) & (channels[None, :] < DIM)
        keys = tl.load(Keys + offsets, valid, 0)
        scores = tl.where(times <= position, tl.sum(keys * q[None, :], 1) * SCALE, -float('inf'))
        next_maximum = tl.maximum(maximum, tl.max(scores, 0))
        carry = tl.exp(maximum - next_maximum)
        probabilities = tl.exp(scores - next_maximum)
        values = tl.load(Values + offsets, valid, 0)
        numerator = numerator * carry + tl.sum(probabilities[:, None] * values, 0)
        denominator = denominator * carry + tl.sum(probabilities, 0)
        maximum = next_maximum
    tl.store(Output + head * DIM + channels, numerator / denominator, channels < DIM)


@triton.jit
def gated_product(Gate, Up, Output, WIDTH: tl.constexpr, KIND: tl.constexpr, TILE: tl.constexpr):
    i = tl.program_id(0) * TILE + tl.arange(0, TILE)
    x = tl.load(Gate + i, i < WIDTH, 0)
    if KIND == 0:
        activation = x / (1. + tl.exp(-x))
    elif KIND == 1:
        activation = 0.5 * x * (1. + libdevice.erf(x * 0.7071067811865476))
    elif KIND == 2:
        activation = tl.maximum(x, 0.)
    else:
        activation = 0.5 * x * (1. + libdevice.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
    tl.store(Output + i, activation * tl.load(Up + i, i < WIDTH, 0), i < WIDTH)


@triton.jit
def vector_sum(X, Y, Output, factor, WIDTH: tl.constexpr, TILE: tl.constexpr):
    i = tl.program_id(0) * TILE + tl.arange(0, TILE)
    tl.store(Output + i, tl.load(X + i, i < WIDTH, 0) + factor * tl.load(Y + i, i < WIDTH, 0), i < WIDTH)


@triton.jit
def plane_rotation(Hidden, U, V, Output, cosine_minus_one, sine,
                   WIDTH: tl.constexpr, TILE: tl.constexpr):
    i = tl.arange(0, TILE)
    h = tl.load(Hidden + i, i < WIDTH, 0)
    u = tl.load(U + i, i < WIDTH, 0)
    v = tl.load(V + i, i < WIDTH, 0)
    hu = tl.sum(h * u, 0)
    hv = tl.sum(h * v, 0)
    changed = h + u * (cosine_minus_one * hu + sine * hv) + v * (cosine_minus_one * hv - sine * hu)
    tl.store(Output + i, changed, i < WIDTH)
