import math


class SingleDropTracker:
    def __init__(
        self,
        max_trajectory=120,
        max_jump=55,
        init_point=None,
        max_miss=12,
        base_radius=80,
        max_radius=140
    ):
        self.max_trajectory = max_trajectory
        self.max_jump = max_jump
        self.max_miss = max_miss
        self.base_radius = base_radius
        self.max_radius = max_radius

        # 轨迹点: (x, y, measured)
        self.trajectory = []
        self.last_point = None
        self.last_radius = None

        self.vx = 0.0
        self.vy = 0.0
        self.missed = 0

        if init_point is not None:
            x, y = init_point
            self.last_point = (int(x), int(y))
            self.trajectory.append((int(x), int(y), True))

    def predict(self):
        if self.last_point is None:
            return None

        x, y = self.last_point
        px = int(round(x + self.vx))
        py = int(round(y + self.vy))
        return px, py

    def get_search_center(self):
        pred = self.predict()
        if pred is not None:
            return pred
        return self.last_point

    def get_search_radius(self):
        return min(self.max_radius, self.base_radius + self.missed * 8)

    def is_lost(self):
        return self.missed > self.max_miss

    def _append_point(self, x, y, measured):
        self.trajectory.append((int(x), int(y), bool(measured)))
        if len(self.trajectory) > self.max_trajectory:
            self.trajectory = self.trajectory[-self.max_trajectory:]

    def _select_target(self, drops):
        if not drops:
            return None

        if self.last_point is None:
            return max(drops, key=lambda d: d[2])

        pred = self.predict()
        if pred is None:
            pred = self.last_point

        px, py = pred

        def cost(drop):
            x, y, r = drop
            dist = math.hypot(x - px, y - py)

            radius_penalty = 0.0
            if self.last_radius is not None:
                radius_penalty = abs(r - self.last_radius) * 3.0

            dx = abs(x - px)
            dy = abs(y - py)

            direction_penalty = dx * 0.6 + dy * 0.2
            return dist + radius_penalty + direction_penalty

        target = min(drops, key=cost)

        tx, ty, tr = target
        dist = math.hypot(tx - px, ty - py)
        allowed_jump = self.max_jump + self.missed * 6

        if dist > allowed_jump:
            return None

        return target

    def update(self, drops):
        if len(drops) == 0:
            self.missed += 1

            pred = self.predict()
            if pred is not None and self.missed <= self.max_miss:
                self.last_point = pred

            return None, self.trajectory

        target = self._select_target(drops)

        if target is None:
            self.missed += 1

            pred = self.predict()
            if pred is not None and self.missed <= self.max_miss:
                self.last_point = pred

            return None, self.trajectory

        x, y, r = target
        x = int(x)
        y = int(y)
        r = int(r)

        if self.last_point is not None:
            lx, ly = self.last_point
            dx = x - lx
            dy = y - ly

            self.vx = 0.72 * self.vx + 0.28 * dx
            self.vy = 0.72 * self.vy + 0.28 * dy

        self.last_point = (x, y)
        self.last_radius = r
        self.missed = 0

        self._append_point(x, y, True)
        return (x, y, r), self.trajectory

    def predict_only(self):
        """
        经过网格线或短时丢失时只更新预测位置，
        但不把预测点写入真实轨迹。
        """
        self.missed += 1

        pred = self.predict()
        if pred is not None and self.missed <= self.max_miss:
            x, y = pred
            self.last_point = (int(x), int(y))

        return None, self.trajectory