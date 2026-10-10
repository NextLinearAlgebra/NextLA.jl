# The split point every recursive routine uses: the largest power of two below
# n, or n ÷ 2 when n is itself a power of two. Power-of-two leading blocks keep
# the kernels on aligned tile sizes, and one rule means the dense drivers and the
# mixed-precision containers cut a matrix the same way. Callers recurse only
# for n >= 2.
@inline _rec_split(n::Integer) = ispow2(n) ? n ÷ 2 : prevpow(2, n)
