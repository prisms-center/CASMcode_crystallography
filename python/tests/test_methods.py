import libcasm.xtal as xtal


def test_make_vacancy():
    vacancy = xtal.make_vacancy()
    assert isinstance(vacancy, xtal.Occupant)
    assert vacancy.name() == "Va"


def test_make_atom():
    atom = xtal.make_atom("A")
    assert isinstance(atom, xtal.Occupant)
    assert atom.name() == "A"
