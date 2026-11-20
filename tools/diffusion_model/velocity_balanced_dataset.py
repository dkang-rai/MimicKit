import torch

from motion_prior_dataset import MotionPriorData

# Bin edges over the last frame's local root velocity: forward vx, sideways vy (m/s) and yaw rate (rad/s).
VX_EDGES = [-0.5, -0.15, 0.15, 0.5, 1.0, 1.5]
VY_EDGES = [-0.4, -0.15, 0.15, 0.4]
YAW_EDGES = [-1.0, -0.3, 0.3, 1.0]


class VelocityBalancedMotionPriorData(MotionPriorData):
    """Resamples windows by inverse frequency of their (vx, vy, yaw rate) bin.

    The pool holds every window of every clip (one per control step), so the dataset
    clip weights are ignored and normalizer statistics cover the entire dataset.
    """

    def __init__(self, config, device):
        super().__init__(config, device)
        assert not self._disc_dof_vel_obs, \
            "Velocity balancing assumes the last 6 features of a frame are root lin/ang vel"

        self._alpha = config.get("velocity_balance_alpha", 0.5)

        self._pool = self._build_full_pool()
        self._weights = self._calc_weights(self._pool)
        return

    @torch.no_grad()
    def _build_full_pool(self, chunk_size=20_000):
        num_motions = self._motion_lib.get_num_motions()
        lengths = self._motion_lib.get_motion_length(torch.arange(num_motions, device=self._device))

        pool = []
        for motion_id in range(num_motions):
            times = torch.arange(0.0, lengths[motion_id].item(), self._timestep, device=self._device)
            ids = torch.full([times.shape[0]], motion_id, dtype=torch.long, device=self._device)
            for i in range(0, times.shape[0], chunk_size):
                pool.append(self._compute_smp_obs_demo(ids[i:i + chunk_size], times[i:i + chunk_size]))
        return torch.cat(pool, dim=0)

    def fetch_stat_obs(self, num_samples):
        return self._pool

    @torch.no_grad()
    def fetch_smp_obs_demo(self, num_samples, return_id=False):
        ids = torch.multinomial(self._weights, num_samples=num_samples, replacement=True)
        return self._pool[ids]

    def _calc_weights(self, pool):
        num_steps = self._num_disc_obs_steps
        last = pool.reshape(pool.shape[0], num_steps, -1)[:, -1]
        vx, vy, yaw = last[:, -6], last[:, -5], last[:, -1]

        bin_ids = torch.zeros_like(vx, dtype=torch.long)
        for value, edges in [(vx, VX_EDGES), (vy, VY_EDGES), (yaw, YAW_EDGES)]:
            edges = torch.tensor(edges, device=pool.device)
            bin_ids = bin_ids * (edges.shape[0] + 1) + torch.bucketize(value, edges)

        num_bins = (len(VX_EDGES) + 1) * (len(VY_EDGES) + 1) * (len(YAW_EDGES) + 1)
        counts = torch.bincount(bin_ids, minlength=num_bins).float()
        weights = counts[bin_ids].pow(-self._alpha)
        return weights / weights.sum()


def build_dataset(config, device):
    sampler = config.get("sampler", "default")
    if sampler == "velocity_balanced":
        return VelocityBalancedMotionPriorData(config, device)
    assert sampler == "default", "Unknown sampler: {}".format(sampler)
    return MotionPriorData(config, device)
