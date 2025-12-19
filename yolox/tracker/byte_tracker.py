import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F
from typing import Iterable, List, Optional, Sequence, Tuple

from .kalman_filter import KalmanFilter
from yolox.tracker import matching
from .basetrack import BaseTrack, TrackState

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    ENV_INFO_KEYS = [
        "road_overlap",
        "nearest_obstacle_px",
        "nearest_obstacle_m",
        "distance_to_wall_m",
        "depth_m",
        "ground_xy",
        "timestamp",
    ]

    def __init__(self, tlwh, score, env_info=None):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
        self.env_info = self._normalize_env_info(env_info)
        self.state_mode = getattr(self.shared_kalman, "state_mode", "2d")

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[self.kalman_filter._velocity_slice] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][st.kalman_filter._velocity_slice] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self._build_measurement(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self._build_measurement(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.env_info = copy.deepcopy(new_track.env_info)

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self._build_measurement(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.env_info = copy.deepcopy(new_track.env_info)

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self._build_measurement(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)

    @classmethod
    def _normalize_env_info(cls, env_info):
        normalized = {key: None for key in cls.ENV_INFO_KEYS}
        if env_info is None or not isinstance(env_info, dict):
            return normalized
        for key in cls.ENV_INFO_KEYS:
            if key in env_info:
                normalized[key] = env_info[key]
        return normalized

    def _build_measurement(self, tlwh):
        base = self.tlwh_to_xyah(tlwh)
        extras = []
        depth, ground_xy = self._extract_3d_components()
        if self.state_mode in ("depth", "z"):
            extras.append(0.0 if depth is None else depth)
        elif self.state_mode == "ground":
            gx, gy = (ground_xy or (0.0, 0.0))
            extras.extend([gx, gy])
            extras.append(0.0 if depth is None else depth)
        if not extras:
            return base
        return np.concatenate([base, np.asarray(extras, dtype=np.float)])

    def _extract_3d_components(self):
        if not isinstance(self.env_info, dict):
            return None, None
        depth = self.env_info.get("depth_m")
        ground_xy = self.env_info.get("ground_xy")
        if ground_xy is not None:
            ground_xy = tuple(ground_xy)
        return depth, ground_xy


class BYTETracker(object):
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        #self.det_thresh = args.track_thresh
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.state_mode = getattr(args, "track_3d", "off")
        if self.state_mode == "off":
            self.state_mode = "2d"
        elif self.state_mode == "z":
            self.state_mode = "depth"
        self.depth_weight = getattr(args, "depth_weight", 0.0)
        self.depth_gate = getattr(args, "depth_gate", None)
        self.project_ground = getattr(args, "project_ground", False) or self.state_mode == "ground"
        self.kalman_filter = KalmanFilter(state_mode=self.state_mode)
        STrack.shared_kalman = self.kalman_filter

    @staticmethod
    def _prepare_env_contexts(env_contexts, length):
        if env_contexts is None:
            return [None] * length
        env_list = list(env_contexts)
        if len(env_list) < length:
            env_list.extend([None] * (length - len(env_list)))
        return env_list[:length]

    @staticmethod
    def _select_env_contexts(env_contexts, mask):
        true_count = int(np.count_nonzero(mask))
        if env_contexts is None:
            return [None] * true_count
        return [
            BYTETracker._extract_env_info(env_contexts[i]) if i < len(env_contexts) else None
            for i, keep in enumerate(mask) if keep
        ]

    @staticmethod
    def _extract_env_info(env_context):
        if env_context is None:
            return None
        if isinstance(env_context, dict) and "env_info" in env_context:
            return env_context.get("env_info")
        return env_context

    @staticmethod
    def _parse_intrinsics(camera_intrinsics):
        if camera_intrinsics is None:
            return None
        matrix = np.asarray(camera_intrinsics, dtype=np.float32)
        if matrix.size == 4:
            fx, fy, cx, cy = matrix.reshape(-1)
        elif matrix.size >= 9:
            matrix = matrix.reshape(3, 3)
            fx, fy, cx, cy = matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]
        else:
            return None
        return float(fx), float(fy), float(cx), float(cy)

    def _compute_depth_env_infos(self, bboxes, img_info, frame_meta):
        if frame_meta is None:
            return [None] * len(bboxes)
        depth_map = frame_meta.get("depth_map") if isinstance(frame_meta, dict) else None
        if depth_map is None:
            return [None] * len(bboxes)
        camera_intrinsics = self._parse_intrinsics(frame_meta.get("camera_intrinsics"))
        project_ground = frame_meta.get("project_ground", self.project_ground) if isinstance(frame_meta, dict) else self.project_ground
        depth_scale = frame_meta.get("depth_scale", 1.0) if isinstance(frame_meta, dict) else 1.0
        depth_map = np.asarray(depth_map)
        if depth_map.ndim == 3:
            depth_map = depth_map[:, :, 0]
        if depth_map.size == 0:
            return [None] * len(bboxes)

        map_h, map_w = depth_map.shape[:2]
        img_h, img_w = img_info[0], img_info[1]
        scale_x = map_w / float(img_w)
        scale_y = map_h / float(img_h)

        env_infos = []
        for bbox in bboxes:
            cx = (bbox[0] + bbox[2]) / 2.0
            foot_y = bbox[3]
            sample_x = int(np.clip(round(cx * scale_x), 0, map_w - 1))
            sample_y = int(np.clip(round(foot_y * scale_y), 0, map_h - 1))

            patch = depth_map[max(0, sample_y - 1):min(map_h, sample_y + 2), max(0, sample_x - 1):min(map_w, sample_x + 2)]
            valid = patch[np.isfinite(patch) & (patch > 0)]
            if valid.size > 0:
                depth_val = float(np.median(valid))
            else:
                depth_val = float(depth_map[sample_y, sample_x])
            depth_val *= depth_scale
            if not np.isfinite(depth_val) or depth_val <= 0:
                env_infos.append(None)
                continue

            ground_xy = None
            if camera_intrinsics is not None and project_ground:
                fx, fy, cx_intr, cy_intr = camera_intrinsics
                ground_x = (cx - cx_intr) * depth_val / fx
                ground_y = (foot_y - cy_intr) * depth_val / fy
                ground_xy = (ground_x, ground_y)

            env_infos.append({"depth_m": depth_val, "ground_xy": ground_xy})
        return env_infos

    def _merge_env_info_contexts(self, env_contexts, computed_env_infos):
        if computed_env_infos is None:
            return env_contexts
        if env_contexts is None:
            env_contexts = [None] * len(computed_env_infos)
        merged = []
        max_len = max(len(env_contexts), len(computed_env_infos))
        for idx in range(max_len):
            base_env = self._extract_env_info(env_contexts[idx]) if idx < len(env_contexts) else None
            computed_env = computed_env_infos[idx] if idx < len(computed_env_infos) else None
            if base_env is None and computed_env is None:
                merged.append(None)
                continue
            combined = {} if base_env is None else dict(base_env)
            if computed_env is not None:
                for key, value in computed_env.items():
                    if value is not None:
                        combined[key] = value
            merged.append(combined)
        return merged

    def _apply_depth_cost(self, cost_matrix, tracks, detections):
        if cost_matrix.size == 0 or self.depth_weight <= 0:
            return cost_matrix
        depth_cost = matching.depth_cost(
            tracks, detections, gate=self.depth_gate, prefer_3d=self.state_mode == "ground"
        )
        return matching.fuse_depth_cost(cost_matrix, depth_cost, weight=self.depth_weight, gate=self.depth_gate)

    def update(self, output_results, img_info, img_size, env_contexts=None, frame_meta=None):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale
        env_contexts = self._prepare_env_contexts(env_contexts, len(bboxes))
        computed_envs = self._compute_depth_env_infos(bboxes, [img_h, img_w], frame_meta)
        env_contexts = self._merge_env_info_contexts(env_contexts, computed_envs)

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        env_keep = self._select_env_contexts(env_contexts, remain_inds)
        env_second = self._select_env_contexts(env_contexts, inds_second)

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, env_info=env) for
                          (tlbr, s, env) in zip(dets, scores_keep, env_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)
        dists = self._apply_depth_cost(dists, strack_pool, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, env_info=env) for
                          (tlbr, s, env) in zip(dets_second, scores_second, env_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        dists = self._apply_depth_cost(dists, r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        dists = self._apply_depth_cost(dists, unconfirmed, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
