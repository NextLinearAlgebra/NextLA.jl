using Test
using KernelAbstractions: CPU
using NextLA: workgroup_reduce!, panel_allreduce!, DeviceParams, verify_budget,
	compute_params, probe_device, panel_cu_set, block_owner

# CPU + any functional GPU backends (same as full `runtests.jl` discovery).
include(joinpath(@__DIR__, "gpu_backends.jl"))

@testset "probe_device / compute_params" begin
	P, M = probe_device(CPU(), Float64)
	@test P >= 1
	@test M >= 1

	p0 = compute_params(CPU(), Float64, 0)
	@test p0.b == 0 && p0.c == 1 && p0.TILE_DIM == 0 && p0.b_max == 0
	@test p0.AI_target == 0.0

	p_neg = compute_params(CPU(), Float64, -3)
	@test p_neg.b == 0 && p_neg.c == 1

	Ntest = 1024
	# Force c=1 so b_min=1 and b lies in [b_min, b_max] on large-memory machines.
	p = compute_params(CPU(), Float64, Ntest; c = 1)
	@test p.c == 1
	@test p.Pz == p.c
	@test p.Py == p.Px
	@test p.P1 == max(1, p.P ÷ p.c)
	@test p.Px == max(1, isqrt(p.P1))
	@test p.b_min == p.c
	@test p.b_max == Ntest ÷ p.Px
	tile_exp = max(1, floor(Int, sqrt(float(p.M))))
	@test p.TILE_DIM == tile_exp
	ai_exp = (2 / 3) * sqrt(float(p.M))
	@test p.AI_target ≈ ai_exp rtol = 1e-6 atol = 1e-6
	@test p.b_min <= p.b <= p.b_max

	cfix = 3
	pc = compute_params(CPU(), Float64, Ntest; c = cfix)
	@test pc.c == cfix
	@test pc.Pz == cfix
	@test pc.P1 == max(1, pc.P ÷ cfix)
	@test pc.Px == max(1, isqrt(pc.P1))
	@test pc.b_min == cfix

	pb = compute_params(CPU(), Float64, Ntest; c = 1, b = p.b)
	@test pb.b == p.b
end

@testset "panel_cu_set / block_owner" begin
	# P=6, c=2 → P1=3, Px=Py=isqrt(3)=1, Pz=2
	params = DeviceParams(6, 100, 2, 2, 3, 1, 1, 2, 10, 2, 200, 1.0)
	@test params.Pz == params.c == 2
	@test params.Px * params.Pz == 2

	for k in (0, 1, 17)
		coords = panel_cu_set(k, params)
		@test length(coords) == params.Px * params.Pz
		j_k = mod(Int(k), params.Py)
		for (r, j, z) in coords
			@test 0 <= r < params.Px
			@test j == j_k
			@test 0 <= z < params.Pz
		end
	end

	I, J = 3, 4
	own = block_owner((I, J), params)
	@test length(own) == params.Pz
	r0 = mod(I, params.Px)
	j0 = mod(J, params.Py)
	for (idx, t) in enumerate(own)
		@test t == (r0, j0, idx - 1)
	end
end

@testset "workgroup_reduce!" begin
	out = [0.0]
	workgroup_reduce!(out, Float64[1, 2, 3, 4]; N = 8)
	@test out[1] == 10.0
	out2 = [0.0]
	workgroup_reduce!(out2, ones(Float64, 256); N = 256)
	@test out2[1] == 256.0
	@test_throws ArgumentError workgroup_reduce!(out2, ones(3); N = 7)
	@test_throws ArgumentError workgroup_reduce!(out2, ones(300); N = 256)
	@test_throws ArgumentError workgroup_reduce!(out2, ones(4); op = *)

	out_one = [0.0]
	workgroup_reduce!(out_one, [5.0]; N = 1)
	@test out_one[1] == 5.0

	@test_throws ArgumentError workgroup_reduce!(out2, ones(4); N = 6)

	empty_out = Float64[]
	@test_throws ArgumentError workgroup_reduce!(empty_out, ones(4); N = 4)

	out_i = [0]
	@test_throws ArgumentError workgroup_reduce!(out_i, Float64[1.0, 2.0]; N = 2)

	out_d = [0.0]
	src_d = Float64[1.0, 2.0, 3.0]
	workgroup_reduce!(out_d, src_d; N = 4)
	workgroup_reduce!(out_d, src_d; N = 4)
	@test out_d[1] == 6.0
end

@testset "panel_allreduce!" begin
	p1 = DeviceParams(4, 100, 2, 1, 4, 2, 2, 1, 8, 1, 50, 1.0)
	G = fill(3.14, 2, 2)
	panel_allreduce!(G, zeros(2, 2, 2), p1)
	@test G[1, 1] == 3.14
	p2 = DeviceParams(8, 1000, 2, 2, 4, 2, 2, 2, 16, 2, 50, 1.0)
	part = reshape(collect(1.0:1.0:16.0), 2, 2, 4)
	G2 = zeros(2, 2)
	panel_allreduce!(G2, part, p2)
	want = dropdims(sum(part, dims = 3), dims = 3)
	@test G2 ≈ want

	p_b0 = DeviceParams(4, 100, 0, 2, 2, 1, 1, 2, 8, 2, 50, 1.0)
	G0 = fill(1.25, 1, 1)
	panel_allreduce!(G0, zeros(1, 1, 2), p_b0)
	@test G0[1, 1] == 1.25

	@test_throws ArgumentError panel_allreduce!(zeros(3, 3), part, p2)
	@test_throws ArgumentError panel_allreduce!(G2, zeros(2, 2, 3), p2)
	Gf = zeros(Float32, 2, 2)
	@test_throws ArgumentError panel_allreduce!(Gf, part, p2)

	p_tree = DeviceParams(10000, 100, 2, 33, 303, 32, 32, 33, 10, 33, 60, 1.0)
	Kbig = p_tree.Px * p_tree.Pz
	@test nextpow(2, Kbig) > 1024
	part_big = zeros(2, 2, Kbig)
	@test_throws ArgumentError panel_allreduce!(zeros(2, 2), part_big, p_tree)

	part_pad = zeros(2, 2, 8)
	part_pad[:, :, 1:4] .= part
	part_pad[:, :, 5] .= 999.0
	Gpad = zeros(2, 2)
	panel_allreduce!(Gpad, part_pad, p2)
	@test Gpad ≈ want

	part32 = Float32.(part)
	G32 = zeros(Float32, 2, 2)
	panel_allreduce!(G32, part32, p2)
	@test G32 ≈ Float32.(want)
end

for (backend_name, ArrayType, synchronize) in available_backends()
	@testset "xpartition kernels [$backend_name]" begin
		@testset "workgroup_reduce!" begin
			out = ArrayType(zeros(Float64, 1))
			src = ArrayType(Float64[1, 2, 3, 4])
			workgroup_reduce!(out, src; N = 8)
			synchronize(out)
			@test Array(out)[1] == 10.0

			out2 = ArrayType(zeros(Float64, 1))
			src2 = ArrayType(ones(Float64, 256))
			workgroup_reduce!(out2, src2; N = 256)
			synchronize(out2)
			@test Array(out2)[1] == 256.0
		end

		@testset "panel_allreduce!" begin
			p2 = DeviceParams(8, 1000, 2, 2, 4, 2, 2, 2, 16, 2, 50, 1.0)
			part = reshape(collect(1.0:1.0:16.0), 2, 2, 4)
			part_dev = ArrayType(part)
			G2 = ArrayType(zeros(2, 2))
			panel_allreduce!(G2, part_dev, p2)
			synchronize(G2)
			want = dropdims(sum(part, dims = 3), dims = 3)
			@test Array(G2) ≈ want

			part_pad = zeros(2, 2, 8)
			part_pad[:, :, 1:4] .= part
			part_pad[:, :, 5] .= 999.0
			part_pad_dev = ArrayType(part_pad)
			Gpad = ArrayType(zeros(2, 2))
			panel_allreduce!(Gpad, part_pad_dev, p2)
			synchronize(Gpad)
			@test Array(Gpad) ≈ want
		end
	end
end

@testset "verify_budget" begin
	params = DeviceParams(8, 1000, 4, 2, 4, 2, 2, 2, 31, 2, 26, 1.0)
	verify_budget(params)
	verify_budget(params; N = 52)

	bad_pz = DeviceParams(4, 100, 2, 1, 4, 2, 2, 99, 8, 1, 50, 1.0)
	@test_throws ArgumentError verify_budget(bad_pz)

	@test_throws ArgumentError verify_budget(DeviceParams(4, 100, 2, 1, 4, 2, 3, 1, 8, 1, 50, 1.0))
	@test_throws ArgumentError verify_budget(DeviceParams(4, 100, 2, 1, 99, 2, 2, 1, 8, 1, 50, 1.0))
	@test_throws ArgumentError verify_budget(DeviceParams(4, 100, 2, 1, 4, 3, 2, 1, 8, 1, 50, 1.0))
	@test_throws ArgumentError verify_budget(DeviceParams(4, 100, 2, 1, 4, 2, 2, 1, 8, 99, 50, 1.0))
	@test_throws ArgumentError verify_budget(DeviceParams(4, 100, 99, 1, 4, 2, 2, 1, 8, 1, 50, 1.0))
	@test_throws ArgumentError verify_budget(DeviceParams(4, 100, 2, 1, 4, 2, 2, 1, 8, 1, 50, 1.0); N = 0)
	p_big = DeviceParams(4, 100, 2, 1, 4, 40, 40, 1, 8, 1, 50, 1.0)
	@test_throws ArgumentError verify_budget(p_big)

	ok_fallback = DeviceParams(2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 10, 1.0)
	verify_budget(ok_fallback; N = 10)
	@test_throws ArgumentError verify_budget(DeviceParams(2, 2, 1, 2, 1, 1, 1, 2, 1, 2, 10, 1.0); N = 10)

	@test_throws ArgumentError verify_budget(params; N = 99)
	@test_throws ArgumentError verify_budget(DeviceParams(8, 1000, 4, 3, 4, 2, 2, 2, 31, 2, 26, 1.0); N = 52)
end
