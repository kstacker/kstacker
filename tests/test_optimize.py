from kstacker.optimize import reject_invalid_orbits


def test_reject_invalid_orbits(params, capsys):
    orbital_grid = params.grid.make_2d_grid(("a", "e", "t0"))
    projection_grid = params.grid.make_2d_grid(("omega", "i", "theta_0"))
    assert orbital_grid.shape == (4480, 3)
    assert projection_grid.shape == (10584, 3)

    orbital_grid, projection_grid, valid_proj = reject_invalid_orbits(
        orbital_grid, projection_grid, params.m0
    )
    out = capsys.readouterr().out.splitlines()
    assert "- 47'416'320 orbits before rejection" in out
    assert "- 27'729'409 orbits after rejection" in out
