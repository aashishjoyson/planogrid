import pytest

from core.exceptions import ComponentAlreadyRegisteredError, ComponentNotRegisteredError
from core.registry import Registry, detector_registry, ocr_registry


def test_register_and_get_roundtrip():
    reg: Registry = Registry("widget")

    @reg.register("foo")
    class Foo:
        pass

    assert reg.get("foo") is Foo
    assert reg.available() == ["foo"]
    assert "foo" in reg
    assert "bar" not in reg


def test_duplicate_registration_raises():
    reg: Registry = Registry("widget")

    @reg.register("foo")
    class Foo:
        pass

    with pytest.raises(ComponentAlreadyRegisteredError):

        @reg.register("foo")
        class Bar:
            pass


def test_unknown_name_raises_and_lists_available():
    reg: Registry = Registry("widget")

    @reg.register("foo")
    class Foo:
        pass

    with pytest.raises(ComponentNotRegisteredError, match="foo"):
        reg.get("bar")


def test_lookup_on_empty_registry_says_so():
    reg: Registry = Registry("widget")
    with pytest.raises(ComponentNotRegisteredError, match="none registered"):
        reg.get("anything")


def test_detector_and_ocr_registries_are_independent_instances():
    assert detector_registry is not ocr_registry
