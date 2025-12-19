import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F
from loguru import logger

from .kalman_filter import KalmanFilter
from yolox.tracker import matching
from .reid_encoder import ReIDEncoder
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

    def __init__(self, tlwh, score, env_info=None, feat=None, smooth_feat=None):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
        self.env_info = self._normalize_env_info(env_info)
        self.curr_feat = feat
        self.smooth_feat = smooth_feat
        self.features = [] if feat is None else [feat]

    def predict(self):
        if self.mean is None or self.kalman_filter is None:
            return
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[self.kalman_filter.ndim:] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    vel_start = st.kalman_filter.ndim if st.kalman_filter is not None else STrack.shared_kalman.ndim
                    multi_mean[i][vel_start:] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        measurement = self.get_measurement(self.kalman_filter)
        self.mean, self.covariance = self.kalman_filter.initiate(measurement)

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.update_features(self.curr_feat)

    def re_activate(self, new_track, frame_id, new_id=False):
        measurement = new_track.get_measurement(self.kalman_filter)
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, measurement
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.env_info = copy.deepcopy(new_track.env_info)
        self.update_features(new_track.curr_feat)

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
        measurement = new_track.get_measurement(self.kalman_filter)
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, measurement)
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.env_info = copy.deepcopy(new_track.env_info)
        self.update_features(new_track.curr_feat)

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
        return self.tlwh_to_xyah(self.tlwh)

    def get_measurement(self, kalman_filter=None):
        xyah = self.tlwh_to_xyah(self._tlwh)
        kf = kalman_filter or self.kalman_filter
        if kf is None or not kf.use_3d_state:
            return xyah
        depth_value = None if self.env_info is None else self.env_info.get("depth_m")
        if depth_value is None:
            logger.debug("3D state requested but depth is missing; using zero depth placeholder.")
            depth_value = 0.0
        return np.append(xyah, depth_value)

    @property
    def depth(self):
        if self.mean is not None and self.kalman_filter is not None and self.kalman_filter.use_3d_state:
            return float(self.mean[4])
        if self.env_info and self.env_info.get("depth_m") is not None:
            return float(self.env_info.get("depth_m"))
        return None

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

    def update_features(self, feat, alpha=0.9):
        if feat is None:
            return
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = alpha * self.smooth_feat + (1 - alpha) * feat
        feat_norm = np.linalg.norm(self.smooth_feat) + 1e-12
        self.smooth_feat = self.smooth_feat / feat_norm
        self.curr_feat = feat
        self.features.append(feat)


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
        self.base_buffer_size = self.buffer_size
        self.max_buffer_scale = getattr(args, "max_buffer_scale", 2.0)
        self.min_match_thresh = getattr(args, "min_match_thresh", 0.1)
        self.buffer_crowd_weight = getattr(args, "buffer_crowd_weight", 1.0)
        self.buffer_road_weight = getattr(args, "buffer_road_weight", 0.5)
        self.match_relax_weight = getattr(args, "match_relax_weight", 0.5)
        self.occluder_grace_frames = getattr(args, "occluder_grace_frames", self.buffer_size)
        self.obstacle_px_threshold = getattr(args, "obstacle_px_threshold", 32.0)
        self.obstacle_m_threshold = getattr(args, "obstacle_m_threshold", 1.0)
        self.wall_proximity_threshold = getattr(args, "wall_proximity_threshold", 1.0)
        self.max_time_lost = self.buffer_size
        self.requested_3d_state = getattr(args, "use_3d_state", False)
        self.depth_weight = getattr(args, "depth_weight", 0.0)
        self.depth_stride = getattr(args, "depth_stride", 1)
        self.depth_gate = 5.0
        self.appearance_fuse = getattr(args, "appearance_fuse", False)
        self.appearance_weight = getattr(args, "appearance_weight", 0.5)
        self.reid_model_path = getattr(args, "reid_model", None)
        self.reid_input_size = getattr(args, "reid_input_size", 128)
        self._reid_encoder = None
        self._active_3d = False
        self._mode_initialized = False
        self._mode_logged = False
        self.kalman_filter = KalmanFilter(use_3d_state=False)
        STrack.shared_kalman = KalmanFilter(use_3d_state=False)

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
    def _extract_intrinsics(img_info):
        if not isinstance(img_info, dict):
            return None
        intrinsics = None
        for key in ["intrinsics", "camera_intrinsics", "cam_intrinsic", "K"]:
            if key in img_info:
                intrinsics = img_info.get(key)
                break
        if intrinsics is None:
            return None
        try:
            intrinsics_array = np.asarray(intrinsics)
            if intrinsics_array.shape == (3, 3):
                fx, fy, cx, cy = intrinsics_array[0, 0], intrinsics_array[1, 1], intrinsics_array[0, 2], intrinsics_array[1, 2]
            elif intrinsics_array.size == 4:
                fx, fy, cx, cy = intrinsics_array.flatten().tolist()
            elif isinstance(intrinsics, dict):
                fx, fy, cx, cy = (
                    intrinsics.get("fx"),
                    intrinsics.get("fy"),
                    intrinsics.get("cx"),
                    intrinsics.get("cy"),
                )
            else:
                return None
            if None in [fx, fy, cx, cy]:
                return None
            return {"fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy)}
        except Exception:
            return None

    @staticmethod
    def _parse_img_info(img_info):
        depth_map = None
        depth_stride = None
        depth_scale = 1.0
        intrinsics = None
        if isinstance(img_info, dict):
            img_h = img_info.get("height", None if "raw_img" not in img_info else img_info["raw_img"].shape[0])
            img_w = img_info.get("width", None if "raw_img" not in img_info else img_info["raw_img"].shape[1])
            depth_map = img_info.get("depth_map", img_info.get("depth"))
            depth_stride = img_info.get("depth_stride")
            depth_scale = img_info.get("depth_scale", 1.0)
            intrinsics = BYTETracker._extract_intrinsics(img_info)
        else:
            img_h, img_w = img_info[0], img_info[1]
        return img_h, img_w, depth_map, intrinsics, depth_stride, depth_scale

    @staticmethod
    def _merge_env_dicts(primary, secondary):
        if primary is None and secondary is None:
            return None
        merged = {}
        for item in [primary, secondary]:
            if item is None:
                continue
            if isinstance(item, dict) and "env_info" in item:
                merged.update(item.get("env_info", {}))
            elif isinstance(item, dict):
                merged.update(item)
        return merged

    @staticmethod
    def _merge_env_lists(primary_list, secondary_list, mask):
        selected_primary = BYTETracker._select_env_contexts(primary_list, mask)
        selected_secondary = BYTETracker._select_env_contexts(secondary_list, mask)
        merged = []
        for primary, secondary in zip(selected_primary, selected_secondary):
            merged.append(BYTETracker._merge_env_dicts(primary, secondary))
        return merged

    @staticmethod
    def _mean_pairwise_iou(tracks):
        active_tracks = [track for track in tracks if track.state in (TrackState.Tracked, TrackState.Lost)]
        if len(active_tracks) < 2:
            return 0.0
        tlbrs = [track.tlbr for track in active_tracks]
        iou_matrix = matching.ious(tlbrs, tlbrs)
        if iou_matrix.size == 0:
            return 0.0
        upper = np.triu_indices(len(active_tracks), k=1)
        pairwise = iou_matrix[upper]
        if pairwise.size == 0:
            return 0.0
        return float(np.mean(pairwise))

    @staticmethod
    def _road_density(tracks, env_contexts):
        densities = []
        for track in tracks:
            env = getattr(track, "env_info", None)
            if env is None:
                continue
            road_overlap = env.get("road_overlap")
            if road_overlap is not None:
                densities.append(float(road_overlap))
        if env_contexts:
            for ctx in env_contexts:
                env = BYTETracker._extract_env_info(ctx)
                if env is None:
                    continue
                road_overlap = env.get("road_overlap")
                if road_overlap is not None:
                    densities.append(float(road_overlap))
        if not densities:
            return 0.0
        return float(np.mean(densities))

    def _compute_buffer_scale(self, crowding_iou, road_density):
        crowd_term = self.buffer_crowd_weight * crowding_iou
        road_term = self.buffer_road_weight * road_density
        scale = 1.0 + crowd_term + road_term
        return min(scale, self.max_buffer_scale)

    def _compute_matching_threshold(self, base_thresh, scale_factor):
        adjusted = base_thresh * (1.0 + self.match_relax_weight * (scale_factor - 1.0))
        return max(self.min_match_thresh, adjusted)

    def _is_near_occluder(self, track):
        env = getattr(track, "env_info", None)
        if env is None:
            return False
        obstacle_px = env.get("nearest_obstacle_px")
        obstacle_m = env.get("nearest_obstacle_m")
        wall_distance = env.get("distance_to_wall_m")
        is_close_px = obstacle_px is not None and obstacle_px <= self.obstacle_px_threshold
        is_close_m = obstacle_m is not None and obstacle_m <= self.obstacle_m_threshold
        is_close_wall = wall_distance is not None and wall_distance <= self.wall_proximity_threshold
        return bool(is_close_px or is_close_m or is_close_wall)

    @staticmethod
    def _log_track_event(track, event, frame_id, details=None):
        message = f"Track {track.track_id} {event} at frame {frame_id}"
        if details:
            message = f"{message} ({details})"
        logger.info(message)

    @staticmethod
    def _depth_for_bbox(tlbr, depth_map, intrinsics, depth_scale=1.0, patch_size=1):
        if depth_map is None or intrinsics is None:
            return None
        if hasattr(depth_map, "detach"):
            depth_arr = depth_map.detach().cpu().numpy()
        else:
            depth_arr = np.asarray(depth_map)
        if depth_arr.ndim == 3:
            depth_arr = depth_arr[..., 0]
        h, w = depth_arr.shape[:2]
        x0, y0, x1, y1 = tlbr
        foot_x = int(round((x0 + x1) / 2.0))
        foot_y = int(round(y1))
        if foot_x < 0 or foot_x >= w or foot_y < 0 or foot_y >= h:
            return None
        window = max(1, int(patch_size))
        half = window // 2
        x_start = max(0, foot_x - half)
        x_end = min(w, foot_x + half + 1)
        y_start = max(0, foot_y - half)
        y_end = min(h, foot_y + half + 1)
        patch = depth_arr[y_start:y_end, x_start:x_end]
        valid = patch[np.isfinite(patch)]
        valid = valid[valid > 0]
        if valid.size == 0:
            return None
        depth_val = float(np.median(valid)) * float(depth_scale)
        fx, fy, cx, cy = intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]
        ground_x = (foot_x - cx) / fx * depth_val
        ground_y = (foot_y - cy) / fy * depth_val
        return {"depth_m": depth_val, "ground_xy": (ground_x, ground_y)}

    @staticmethod
    def _compute_depth_envs(bboxes, depth_map, intrinsics, depth_scale=1.0, patch_size=1):
        if depth_map is None or intrinsics is None:
            return [None] * len(bboxes)
        return [
            BYTETracker._depth_for_bbox(box, depth_map, intrinsics, depth_scale=depth_scale, patch_size=patch_size)
            for box in bboxes
        ]

    def _compute_embeddings(self, image, bboxes):
        if image is None or not self.appearance_fuse:
            return [None] * len(bboxes)
        encoder = self._get_reid_encoder()
        if encoder is None:
            return [None] * len(bboxes)
        embeddings = encoder(image, bboxes)
        if embeddings is None:
            return [None] * len(bboxes)
        return embeddings

    @staticmethod
    def _has_complete_depth(env_infos):
        if not env_infos:
            return False
        for info in env_infos:
            if info is None:
                return False
            if info.get("depth_m") is None:
                return False
        return True

    def _set_motion_model(self, use_3d):
        if use_3d == self._active_3d and self.kalman_filter is not None:
            return
        self._active_3d = use_3d
        self.kalman_filter = KalmanFilter(use_3d_state=use_3d)
        STrack.shared_kalman = KalmanFilter(use_3d_state=use_3d)

    def _get_reid_encoder(self):
        if self._reid_encoder is not None:
            return self._reid_encoder
        if not self.appearance_fuse:
            return None
        try:
            self._reid_encoder = ReIDEncoder(
                device=self.args.device if hasattr(self.args, "device") else None,
                model_path=self.reid_model_path,
                input_size=self.reid_input_size,
            )
        except Exception as exc:
            logger.warning(f"Failed to initialize ReID encoder: {exc}")
            self._reid_encoder = None
        return self._reid_encoder

    def _maybe_initialize_mode(self, depth_map, intrinsics, env_infos):
        if self._mode_initialized:
            return
        activate_3d = bool(self.requested_3d_state and depth_map is not None and intrinsics is not None)
        if env_infos:
            activate_3d = activate_3d and self._has_complete_depth(env_infos)
        self._set_motion_model(activate_3d)
        self._mode_initialized = True
        if self._mode_logged:
            return
        if self._active_3d:
            logger.info("3D-enriched tracking mode enabled (depth + calibration detected).")
        elif self.requested_3d_state:
            logger.info("Depth or calibration missing; defaulting to 2D tracking mode.")
        else:
            logger.info("2D tracking mode active (default).")
        self._mode_logged = True

    def update(self, output_results, img_info, img_size, env_contexts=None):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        raw_img = img_info["raw_img"] if isinstance(img_info, dict) and "raw_img" in img_info else None
        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        img_h, img_w, depth_map, intrinsics, scene_depth_stride, depth_scale = self._parse_img_info(img_info)
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale
        env_contexts = self._prepare_env_contexts(env_contexts, len(bboxes))
        depth_patch = scene_depth_stride if scene_depth_stride is not None else self.depth_stride
        depth_envs = self._compute_depth_envs(bboxes, depth_map, intrinsics, depth_scale=depth_scale, patch_size=depth_patch)
        embeddings = self._compute_embeddings(raw_img, bboxes.tolist()) if len(bboxes) > 0 else []

        crowding_iou = self._mean_pairwise_iou(self.tracked_stracks + self.lost_stracks)
        road_density = self._road_density(self.tracked_stracks + self.lost_stracks, env_contexts)
        buffer_scale = self._compute_buffer_scale(crowding_iou, road_density)
        self.buffer_size = int(round(self.base_buffer_size * buffer_scale))
        self.max_time_lost = self.buffer_size
        primary_match_thresh = self._compute_matching_threshold(self.args.match_thresh, buffer_scale)
        secondary_match_thresh = self._compute_matching_threshold(0.5, buffer_scale)
        unconfirmed_match_thresh = self._compute_matching_threshold(0.7, buffer_scale)
        logger.debug(
            "Frame %s: crowding_iou=%.3f road_density=%.3f buffer_scale=%.2f buffer=%d match_thresh=%.3f",
            self.frame_id,
            crowding_iou,
            road_density,
            buffer_scale,
            self.buffer_size,
            primary_match_thresh,
        )

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        env_keep = self._merge_env_lists(env_contexts, depth_envs, remain_inds)
        env_second = self._merge_env_lists(env_contexts, depth_envs, inds_second)
        feat_keep = [embeddings[i] for i, keep in enumerate(remain_inds) if keep] if embeddings else []
        feat_second = [embeddings[i] for i, keep in enumerate(inds_second) if keep] if embeddings else []

        self._maybe_initialize_mode(depth_map, intrinsics, env_keep + env_second)

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, env_info=env, feat=feat, smooth_feat=feat) for
                          (tlbr, s, env, feat) in zip(dets, scores_keep, env_keep, feat_keep)]
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
        if self.appearance_fuse:
            app_cost = matching.embedding_distance(strack_pool, detections)
            dists = matching.combine_costs(dists, app_cost, self.appearance_weight)
        if self._active_3d and self.depth_weight > 0:
            depth_dists = matching.depth_distance(strack_pool, detections, max_depth_jump=self.depth_gate)
            dists = matching.combine_costs(dists, depth_dists, self.depth_weight)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=primary_match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                self._log_track_event(track, "updated", self.frame_id, details="primary association")
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                self._log_track_event(track, "reactivated", self.frame_id, details="primary association")
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, env_info=env, feat=feat, smooth_feat=feat) for
                          (tlbr, s, env, feat) in zip(dets_second, scores_second, env_second, feat_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        if self.appearance_fuse:
            app_cost = matching.embedding_distance(r_tracked_stracks, detections_second)
            dists = matching.combine_costs(dists, app_cost, self.appearance_weight)
        if self._active_3d and self.depth_weight > 0:
            depth_dists = matching.depth_distance(r_tracked_stracks, detections_second, max_depth_jump=self.depth_gate)
            dists = matching.combine_costs(dists, depth_dists, self.depth_weight)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=secondary_match_thresh)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                self._log_track_event(track, "updated", self.frame_id, details="secondary association")
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                self._log_track_event(track, "reactivated", self.frame_id, details="secondary association")
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                self._log_track_event(track, "lost", self.frame_id)
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        if self.appearance_fuse:
            app_cost = matching.embedding_distance(unconfirmed, detections)
            dists = matching.combine_costs(dists, app_cost, self.appearance_weight)
        if self._active_3d and self.depth_weight > 0:
            depth_dists = matching.depth_distance(unconfirmed, detections, max_depth_jump=self.depth_gate)
            dists = matching.combine_costs(dists, depth_dists, self.depth_weight)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=unconfirmed_match_thresh)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            self._log_track_event(unconfirmed[itracked], "confirmed", self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            self._log_track_event(track, "removed", self.frame_id, details="unconfirmed drop")
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            self._log_track_event(track, "activated", self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            elapsed = self.frame_id - track.end_frame
            grace = self.occluder_grace_frames if self._is_near_occluder(track) else 0
            if elapsed > self.max_time_lost + grace:
                track.mark_removed()
                self._log_track_event(track, "removed", self.frame_id, details="time_exceeded")
                removed_stracks.append(track)
            elif grace > 0:
                self._log_track_event(track, "grace_period", self.frame_id, details=f"elapsed={elapsed}, grace={grace}")

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
