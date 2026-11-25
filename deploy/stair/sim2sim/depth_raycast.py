"""CPU ray queries matching the training Warp BVH's mesh backface rule."""

from copy import copy

import mujoco
import numpy as np


class TrainingRaycaster:
    """Reject a mesh when its nearest intersection is a backface, as Warp does.

    Re-query the affected rays with that geometry excluded, from the original
    origin. Advancing the origin or masking the resulting pixels would miss the
    terrain behind the head. Other robot meshes remain eligible for occlusion.
    A private model keeps temporary ray filters away from physics and the GUI;
    poses still come from the live MjData (with the same model layout).
    """

    def __init__(self, model, groups, bodyexclude=-1):
        self.model = copy(model)
        self.bodyexclude = bodyexclude
        # Reserve group 1 for exclusions, including when all six source groups
        # are enabled. The private model's original group numbers are immaterial.
        self.base_groups = np.where(groups[model.geom_group], 0, 1)
        self.model.geom_group[:] = self.base_groups
        self.groups = np.array((1, 0, 0, 0, 0, 0), dtype=np.uint8)
        self.surface_mesh = np.isin(model.geom_type, (
            mujoco.mjtGeom.mjGEOM_MESH, mujoco.mjtGeom.mjGEOM_HFIELD,
        ))

    def cast(self, data, origin, directions):
        distances = np.full(len(directions), -1.0)
        geom_ids = np.full(len(directions), -1, dtype=np.int32)
        pending = [(np.arange(len(directions)), ())]
        try:
            while pending:
                indices, excluded = pending.pop()
                self.model.geom_group[:] = self.base_groups
                self.model.geom_group[list(excluded)] = 1
                rays = np.ascontiguousarray(directions[indices])
                hits = np.empty(len(indices), dtype=np.int32)
                ranges = np.empty(len(indices))
                normals = np.empty_like(rays)
                # cutoff culls geometry origins, not intersections: distant
                # plane/large-box origins must not remove nearby ground.
                mujoco.mj_multiRay(
                    self.model, data, origin, rays.reshape(-1), self.groups,
                    True, self.bodyexclude, hits, ranges, normals.reshape(-1),
                    len(indices), mujoco.mjMAXVAL,
                )
                backface = ((hits >= 0) & self.surface_mesh[np.maximum(hits, 0)]
                            & (np.einsum("ij,ij->i", rays, normals) >= 0))
                accepted = ~backface
                distances[indices[accepted]] = ranges[accepted]
                geom_ids[indices[accepted]] = hits[accepted]
                for geom in np.unique(hits[backface]):
                    affected = indices[backface & (hits == geom)]
                    pending.append((affected, (*excluded, int(geom))))
        finally:
            self.model.geom_group[:] = self.base_groups
        return distances, geom_ids
