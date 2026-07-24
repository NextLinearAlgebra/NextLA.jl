@testset "row-basis workspace liveness plan" begin
    shape = _TLRM.RowBasisWorkspaceShape(64, 8, 6, 10, 12, 16, 4;
                                    h=2, jblock=3, q=5)
    plan = _TLRM.row_basis_workspace_plan(shape, Float32)
    @test plan.pipeline_slots == 1
    @test plan.bytes == max(plan.basis_bytes, plan.accumulation_bytes, plan.merge_bytes)
    @test _TLRM.workspace_fits(plan, plan.bytes)
    @test !_TLRM.workspace_fits(plan, plan.bytes - 1)

    # The usual compressed branch has no S_ikj allocation; making rB wider
    # affects Rstack but does not alter the A-side T-panel requirement.
    narrow = _TLRM.RowBasisWorkspaceShape(64, 8, 6, 2, 12, 16, 4;
                                     h=2, jblock=3, q=5)
    wide = _TLRM.RowBasisWorkspaceShape(64, 8, 6, 20, 12, 16, 4;
                                   h=2, jblock=3, q=5)
    pnarrow = _TLRM.row_basis_workspace_plan(narrow, Float32)
    pwide = _TLRM.row_basis_workspace_plan(wide, Float32)
    @test pwide.accumulation_bytes > pnarrow.accumulation_bytes

    # A two-slot pipeline must reserve disjoint basis and accumulation/merge
    # arenas; it can never be smaller than the single-stream carved arena.
    pipelined = _TLRM.row_basis_workspace_plan(shape, Float32; pipeline_slots=2)
    @test pipelined.bytes == plan.basis_bytes +
                             max(plan.accumulation_bytes, plan.merge_bytes)
    @test pipelined.bytes >= plan.bytes

    @test_throws ArgumentError _TLRM.RowBasisWorkspaceShape(64, 8, 6, 10, 12, 65, 4)
    @test_throws ArgumentError _TLRM.RowBasisWorkspaceShape(64, 8, 6, 10, 12, 16, 17)
    @test_throws ArgumentError _TLRM.row_basis_workspace_plan(shape, Float32; pipeline_slots=3)
end
