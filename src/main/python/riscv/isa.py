import abc
from typing import Callable, Type

from riscv.decode import insn_t
from riscv.disasm import arg_t
from riscv.extension import extension_t, rocc_t, register_extension


__all__ = ['ISA', 'ROCC', 'register']


class ISA(extension_t):
    """
    ISA Abstract Base
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError("abstract class")

    def _name(self) -> str:
        # C++ -> Python call to extension_t::name() hits this
        return self.name


class ROCC(rocc_t, ISA):
    """
    RoCC Abstract Base
    """


def register(ext_name: str):
    """
    Decorator for registering ISA/RoCC extension
    """

    # if find_extension(name) is not None:
    #     raise KeyError(f"Extension '{name}' already registered")

    def isa_decorator(ext_cls: Type[extension_t]):

        class MyISA(ext_cls):

            @property
            def name(self) -> str:
                return ext_name

        MyISA.__name__ = ext_cls.__name__
        MyISA.__doc__ = ext_cls.__doc__

        register_extension(ext_name, MyISA)
        return MyISA

    return isa_decorator


def arg(func: Callable[[insn_t], str]):
    """
    Decorator for formatting insn operand / arg
    """

    class MyArg(arg_t):

        def to_string(self, insn: insn_t) -> str:
            return func(insn)

    MyArg.__name__ = func.__name__
    MyArg.__doc__ = func.__doc__

    return MyArg()
