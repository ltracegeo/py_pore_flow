import numba as nb
from numba.experimental import jitclass

spec = [
    ("enable_control", nb.types.float64),
    ("cycle", nb.types.int32),
    ("DRAINAGE_CYCLE", nb.types.int32),
    ("IMBIBITION_CYCLE", nb.types.int32),
    ("SECOND_DRAINAGE_CYCLE", nb.types.int32),
    ("pc", nb.types.float64),
    ("pc_step_initial", nb.types.float64),
    ("pc_step", nb.types.float64),
    ("pc_step_min", nb.types.float64),
    ("pc_step_max", nb.types.float64),
    ("sw_step", nb.types.float64),
    ("pc_maximum", nb.types.float64),
    ("pc_minimum", nb.types.float64),
    ("sw_last", nb.types.float64),
    ("first_step", nb.types.boolean),
]


@jitclass(spec)
class PcIncrementer:
    def __init__(
        self,
        pc_step,
        pc_step_min,
        pc_step_max,
        sw_step,
        pc_maximum,
        pc_minimum,
        pc_initial=0,
        enable_control=True,
    ):
        self.DRAINAGE_CYCLE = 0
        self.IMBIBITION_CYCLE = 1
        self.SECOND_DRAINAGE_CYCLE = 2

        self.enable_control = enable_control
        self.pc_step_initial = pc_step
        self.pc_step_min = pc_step_min
        self.pc_step_max = pc_step_max
        self.sw_step = sw_step
        self.pc_maximum = pc_maximum
        self.pc_minimum = pc_minimum

        self.cycle = self.DRAINAGE_CYCLE

        self.pc = pc_initial
        self.pc_step = self.pc_step_initial

        self.first_step = True
        self.sw_last = 0

    def set_cycle(self, cycle):
        self.cycle = cycle

    def get_pc(self):
        return self.pc

    def get_pc_max(self):
        return self.pc_maximum

    def get_pc_min(self):
        return self.pc_minimum

    def set_pc(self, new_pc):
        self.pc = new_pc

    def set_pc_to_min(self):
        self.pc = self.pc_minimum

    def set_pc_to_max(self):
        self.pc = self.pc_maximum

    def increment(self, sw_current):
        self._adjust_pc_step(sw_current)
        self.pc = min(self.pc + self.pc_step, self.pc_maximum)
        self.sw_last = sw_current
        self.first_step = False
        return self.pc

    def decrement(self, sw_current):
        self._adjust_pc_step(sw_current)
        self.pc = max(self.pc - self.pc_step, self.pc_minimum)
        self.sw_last = sw_current
        self.first_step = False
        return self.pc

    def update_pc(self, sw_current=0.0):
        if self.cycle == self.DRAINAGE_CYCLE or self.cycle == self.SECOND_DRAINAGE_CYCLE:
            return self.increment(sw_current)
        else:
            return self.decrement(sw_current)

    def cycle_should_end(self):
        if self.cycle == self.DRAINAGE_CYCLE or self.cycle == self.SECOND_DRAINAGE_CYCLE:
            return self.pc >= self.get_pc_max()
        else:
            return self.pc <= self.get_pc_min()

    def _adjust_pc_step(self, sw_current):
        if self.enable_control and not self.first_step:
            sw_diff = abs(self.sw_last - sw_current)
            if sw_diff > 0:
                error = self.sw_step / sw_diff
                if error > 0:
                    self.pc_step *= error / 2
                else:
                    self.pc_step *= error
            else:
                self.pc_step = self.pc_step_initial
            self.pc_step = max(self.pc_step, self.pc_step_min)
            self.pc_step = min(self.pc_step, self.pc_step_max)
        else:
            self.pc_step = self.pc_step_initial
