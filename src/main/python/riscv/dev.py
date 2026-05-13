from typing import Optional, Tuple, Type

from riscv.devices import abstract_device_t, device_factory_t, mmio_device_map
from riscv.sim import sim_t


__all__ = ['MMIO', 'register']


class MMIO(abstract_device_t):
    """
    MMIO Abstract Base
    """

    def __init__(self, sim: sim_t, args: Optional[str] = None):
        super().__init__()
        self.sim: sim_t = sim
        self.args: Optional[str] = args

    def tick(self, rtc_ticks: int) -> None:
        pass


def register(name: str, *, size: Optional[int] = None, replace: bool = False):
    """
    Decorator for registering MMIO device
    """

    if name in mmio_device_map and not replace:
        raise KeyError(f"Device factory '{name}' already registered")

    def mmio_decorator(mmio_cls: Type[MMIO]):

        class MMIODevice(mmio_cls):

            def size(self) -> int:
                if size is not None:
                    return size
                return super().size()

        MMIODevice.__name__ = mmio_cls.__name__
        MMIODevice.__doc__ = mmio_cls.__doc__

        class MMIOFactory(device_factory_t):

            # pylint: disable=unused-argument
            def parse_from_fdt(self, fdt, sim: sim_t, *sargs: str) -> Tuple[Optional[abstract_device_t], int]:
                return MMIODevice(sim, ",".join(sargs[1:])), int(sargs[0], 16)

            # pylint: disable=unused-argument
            def generate_dts(self, sim: sim_t, *sargs: str) -> str:
                return ""

        mmio_device_map[name] = MMIOFactory()
        return MMIODevice

    return mmio_decorator
