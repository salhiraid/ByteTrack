import argparse
import json
import os
import os.path as osp
import time
import cv2
import numpy as np
import torch

from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
ENV_INFO_KEYS = [
    "road_overlap",
    "nearest_obstacle_px",
    "nearest_obstacle_m",
    "distance_to_wall_m",
    "depth_m",
    "ground_xy",
    "timestamp",
]


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        #"--path", default="./datasets/mot/train/MOT17-05-FRCNN/img1", help="path to images or video"
        "--path", default="./videos/palace.mp4", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    parser.add_argument(
        "--road_gating_mode",
        type=str,
        default="soft",
        choices=["off", "soft", "hard"],
        help="How to apply road-overlap gating: 'soft' scales scores, 'hard' drops, 'off' disables gating.",
    )
    parser.add_argument(
        "--road_overlap_thresh",
        type=float,
        default=0.3,
        help="Minimum road-overlap ratio used by road gating.",
    )
    parser.add_argument(
        "--use_3d_state",
        action="store_true",
        help="Enable optional 3D-enriched Kalman state when depth and calibration inputs are available.",
    )
    parser.add_argument(
        "--depth_weight",
        type=float,
        default=0.5,
        help="Weight for the depth/3D consistency term during matching (0 disables).",
    )
    parser.add_argument(
        "--depth_stride",
        type=int,
        default=1,
        help="Stride/window size for sampling the depth map around detection footpoints. Scene-provided stride overrides this.",
    )
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


def _to_serializable_value(value):
    if value is None:
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_to_serializable_value(v) for v in value]
    return value


def sanitize_env_info(env_info):
    if not env_info:
        return {}
    sanitized = {}
    for key in ENV_INFO_KEYS:
        if key in env_info and env_info[key] is not None:
            sanitized[key] = _to_serializable_value(env_info[key])
    return sanitized


def _extract_road_mask(img_info):
    if not isinstance(img_info, dict):
        return None
    for key in ["road_mask", "segmentation_mask", "seg_mask"]:
        if key in img_info:
            return img_info[key]
    return None


def _prepare_road_mask(img_info):
    raw_mask = _extract_road_mask(img_info)
    if raw_mask is None:
        return None
    try:
        mask = np.array(raw_mask)
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.dtype != np.bool_:
            mask = mask > 0
        raw_img = img_info.get("raw_img")
        if raw_img is not None and mask.shape[:2] != raw_img.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (raw_img.shape[1], raw_img.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask = mask > 0
        return mask
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Failed to prepare road mask, bypassing gating. Error: {exc}")
        return None


def _compute_road_overlap(detections, mask):
    if mask is None or detections.size == 0:
        return None
    mask_h, mask_w = mask.shape[:2]
    overlaps = []
    for det in detections:
        x0, y0, x1, y1 = det[:4]
        x0i, y0i, x1i, y1i = map(int, [x0, y0, x1, y1])
        x0i = np.clip(x0i, 0, mask_w)
        x1i = np.clip(x1i, 0, mask_w)
        y0i = np.clip(y0i, 0, mask_h)
        y1i = np.clip(y1i, 0, mask_h)
        if x1i <= x0i or y1i <= y0i:
            overlaps.append(0.0)
            continue
        region = mask[y0i:y1i, x0i:x1i]
        if region.size == 0:
            overlaps.append(0.0)
            continue
        overlaps.append(float(np.count_nonzero(region)) / float(region.size))
    return np.asarray(overlaps, dtype=np.float32)


def _apply_road_gating(outputs, img_info, args):
    if args.road_gating_mode == "off":
        return outputs, None, None
    if outputs is None or outputs[0] is None or outputs[0].shape[0] == 0:
        return outputs, None, None

    road_mask = _prepare_road_mask(img_info)
    if road_mask is None:
        return outputs, None, {"segmentation_available": False}

    try:
        det_tensor = outputs[0]
        device = det_tensor.device if torch.is_tensor(det_tensor) else None
        dets_np = det_tensor.detach().cpu().numpy()
        overlaps = _compute_road_overlap(dets_np, road_mask)
        if overlaps is None:
            return outputs, None, {"segmentation_available": False}

        threshold = max(args.road_overlap_thresh, 1e-6)
        env_contexts = [{"env_info": {"road_overlap": float(ov)}} for ov in overlaps]

        if args.road_gating_mode == "hard":
            keep_mask = overlaps >= threshold
            gated_np = dets_np[keep_mask]
            env_contexts = [env_contexts[i] for i, keep in enumerate(keep_mask) if keep]
            dropped = int(np.count_nonzero(~keep_mask))
            kept = int(gated_np.shape[0])
            below = int(np.count_nonzero(overlaps < threshold))
        else:
            scale = np.clip(overlaps / threshold, 0.0, 1.0)
            gated_np = dets_np.copy()
            gated_np[:, 4] = gated_np[:, 4] * scale
            keep_mask = np.ones_like(scale, dtype=bool)
            dropped = 0
            kept = int(gated_np.shape[0])
            below = int(np.count_nonzero(overlaps < threshold))
            env_contexts = env_contexts  # keep alignment for soft mode

        if torch.is_tensor(det_tensor):
            gated_tensor = torch.from_numpy(gated_np).to(device=device, dtype=det_tensor.dtype)
        else:
            gated_tensor = gated_np

        gated_outputs = list(outputs)
        gated_outputs[0] = gated_tensor

        stats = {
            "mode": args.road_gating_mode,
            "total": int(len(overlaps)),
            "kept": kept,
            "dropped": dropped,
            "below_threshold": below,
            "segmentation_available": True,
        }
        return gated_outputs, env_contexts, stats
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Road gating failed; continuing without gating. Error: {exc}")
        return outputs, None, {"segmentation_available": False}


def _log_gating_stats(frame_id, stats):
    if not stats:
        return
    if not stats.get("segmentation_available", True):
        logger.debug(f"Frame {frame_id}: segmentation unavailable, skipped road gating.")
        return
    logger.info(
        "Frame {frame}: road gating mode={mode}, total={total}, kept={kept}, dropped={dropped}, below_thresh={below}".format(
            frame=frame_id,
            mode=stats.get("mode"),
            total=stats.get("total"),
            kept=stats.get("kept"),
            dropped=stats.get("dropped"),
            below=stats.get("below_threshold"),
        )
    )


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
            #logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info


def image_demo(predictor, vis_folder, current_time, args):
    if osp.isdir(args.path):
        files = get_image_list(args.path)
    else:
        files = [args.path]
    files.sort()
    tracker = BYTETracker(args, frame_rate=args.fps)
    timer = Timer()
    results = []
    env_info_records = []
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time) if args.save_result else None
    if args.save_result:
        save_folder = osp.join(vis_folder, timestamp)
        os.makedirs(save_folder, exist_ok=True)

    for frame_id, img_path in enumerate(files, 1):
        outputs, img_info = predictor.inference(img_path, timer)
        outputs, env_contexts, gating_stats = _apply_road_gating(outputs, img_info, args)
        if outputs[0] is not None:
            online_targets = tracker.update(
                outputs[0], img_info, exp.test_size, env_contexts=env_contexts
            )
            online_tlwhs = []
            online_ids = []
            online_scores = []
            online_env_infos = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    online_env_infos.append(t.env_info)
                    # save results
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )
                    env_info_records.append(
                        {
                            "frame": frame_id,
                            "id": int(tid),
                            "score": float(t.score),
                            "env_info": sanitize_env_info(t.env_info),
                        }
                    )
            timer.toc()
            online_im = plot_tracking(
                img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id, fps=1. / timer.average_time,
                env_infos=online_env_infos
            )
        else:
            timer.toc()
            online_im = img_info['raw_img']

        _log_gating_stats(frame_id, gating_stats)

        # result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if args.save_result:
            cv2.imwrite(osp.join(save_folder, osp.basename(img_path)), online_im)

        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")
        env_file = osp.join(vis_folder, f"{timestamp}_env_info.json")
        with open(env_file, "w") as f:
            json.dump(env_info_records, f, indent=2)
        logger.info(f"save env info to {env_file}")


def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = osp.join(save_folder, args.path.split("/")[-1])
    else:
        save_path = osp.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    tracker = BYTETracker(args, frame_rate=30)
    timer = Timer()
    frame_id = 0
    results = []
    env_info_records = []
    while True:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame, timer)
            outputs, env_contexts, gating_stats = _apply_road_gating(outputs, img_info, args)
            if outputs[0] is not None:
                online_targets = tracker.update(
                    outputs[0], img_info, exp.test_size, env_contexts=env_contexts
                )
                online_tlwhs = []
                online_ids = []
                online_scores = []
                online_env_infos = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        online_env_infos.append(t.env_info)
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                        env_info_records.append(
                            {
                                "frame": frame_id,
                                "id": int(tid),
                                "score": float(t.score),
                                "env_info": sanitize_env_info(t.env_info),
                            }
                        )
                timer.toc()
                online_im = plot_tracking(
                    img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time,
                    env_infos=online_env_infos
                )
            else:
                timer.toc()
                online_im = img_info['raw_img']
            _log_gating_stats(frame_id + 1, gating_stats)
            if args.save_result:
                vid_writer.write(online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")
        env_file = osp.join(vis_folder, f"{timestamp}_env_info.json")
        with open(env_file, "w") as f:
            json.dump(env_info_records, f, indent=2)
        logger.info(f"save env info to {env_file}")


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    output_dir = osp.join(exp.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    if args.save_result:
        vis_folder = osp.join(output_dir, "track_vis")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = osp.join(output_dir, "model_trt.pth")
        assert osp.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_folder, current_time, args)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
