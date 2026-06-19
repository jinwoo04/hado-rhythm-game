"""YOLOv8 인물 감지 래퍼.

ultralytics YOLO 모델을 단일 클래스(person)에 대해 추론하고,
바운딩 박스 + 신뢰도 + 발 위치(하단 중앙)를 numpy 배열로 반환.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Detection:
    """단일 감지 결과."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    keypoints: "np.ndarray | None" = None  # (17, 3) [x, y, conf] — pose 모델 사용 시만

    @property
    def bbox(self) -> np.ndarray:
        """[x1, y1, x2, y2] 형태."""
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float32)

    @property
    def foot_point(self) -> tuple[float, float]:
        """발 위치 = 바운딩 박스 하단 중앙. Homography 변환 입력."""
        return ((self.x1 + self.x2) / 2.0, self.y2)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


class PersonDetector:
    """YOLOv8 person 감지기.

    사용 예시
    ---------
    >>> det = PersonDetector(model_path="yolov8n.pt", imgsz=320)
    >>> dets = det.detect(frame)
    >>> for d in dets:
    ...     print(d.bbox, d.confidence)
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        imgsz: int = 320,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.5,
        target_class: int = 0,
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.target_class = target_class
        self.device = device

        # Lazy import (테스트/CI 환경에서 ultralytics 없어도 호환)
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as e:
            raise ImportError(
                "ultralytics 패키지가 필요합니다. `pip install ultralytics` 실행하세요."
            ) from e

        print(f"[Detector] 모델 로드 중: {model_path}")
        task = "pose" if "pose" in str(model_path) else "detect"
        self.model = YOLO(model_path, task=task)
        if task == "pose":
            _dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
            try:
                self.model.predict(_dummy, imgsz=imgsz, verbose=False)
            except AttributeError as _e:
                if "kpt_shape" in str(_e):
                    self.model.predictor.model.kpt_shape = [17, 3]
                    self.model.predict(_dummy, imgsz=imgsz, verbose=False)
        print(f"[Detector] 준비 완료 (imgsz={imgsz}, conf={conf_threshold}, device={device})")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """프레임에서 person 감지.

        Parameters
        ----------
        frame : BGR ndarray (H, W, 3)

        Returns
        -------
        list[Detection]
        """
        _kwargs = dict(
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=[self.target_class],
            device=self.device,
            verbose=False,
        )
        try:
            results = self.model(frame, **_kwargs)
        except AttributeError as _e:
            if "kpt_shape" not in str(_e):
                raise
            self.model.predictor.model.kpt_shape = [17, 3]
            results = self.model(frame, **_kwargs)

        detections: list[Detection] = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        # xyxy 좌표 (절대 픽셀)
        xyxy = boxes.xyxy.cpu().numpy()  # (N, 4)
        conf = boxes.conf.cpu().numpy()   # (N,)

        # pose 모델이면 keypoints 추출
        kpts_raw = getattr(results[0], "keypoints", None)
        kpts_data = kpts_raw.data.cpu().numpy() if (kpts_raw is not None and len(kpts_raw) > 0) else None

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            kpts = kpts_data[i] if kpts_data is not None else None  # (17, 3) or None
            detections.append(Detection(
                x1=float(x1), y1=float(y1),
                x2=float(x2), y2=float(y2),
                confidence=float(conf[i]),
                keypoints=kpts,
            ))

        return detections


def main():
    """단독 실행: 이미지 또는 카메라로 감지 테스트."""
    import argparse
    import time

    import cv2

    from src.camera import Camera

    parser = argparse.ArgumentParser(description="YOLOv8 인물 감지 테스트")
    parser.add_argument("--source", default=0, help="0 (카메라) 또는 이미지/비디오 경로")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.35)
    args = parser.parse_args()

    detector = PersonDetector(model_path=args.model, imgsz=args.imgsz, conf_threshold=args.conf)

    # 입력 분기
    try:
        source = int(args.source)
        is_camera = True
    except ValueError:
        source = args.source
        is_camera = False

    if is_camera:
        with Camera(source=source) as cam:
            t_start = time.time()
            frame_count = 0
            while True:
                ok, frame = cam.read()
                if not ok:
                    break
                dets = detector.detect(frame)
                for d in dets:
                    cv2.rectangle(frame, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)),
                                  (0, 255, 0), 2)
                    cv2.putText(frame, f"{d.confidence:.2f}",
                                (int(d.x1), int(d.y1) - 6), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 1)
                    # 발 위치
                    fx, fy = d.foot_point
                    cv2.circle(frame, (int(fx), int(fy)), 5, (0, 0, 255), -1)

                frame_count += 1
                fps = frame_count / max(0.01, time.time() - t_start)
                cv2.putText(frame, f"FPS: {fps:.1f} | persons: {len(dets)}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                cv2.imshow("Detection (ESC to exit)", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            cv2.destroyAllWindows()
    else:
        # 이미지/비디오 파일 경로
        img = cv2.imread(source)
        if img is None:
            print(f"이미지 로드 실패: {source}")
            return
        dets = detector.detect(img)
        print(f"감지된 사람: {len(dets)}")
        for i, d in enumerate(dets):
            print(f"  [{i}] bbox=({d.x1:.0f},{d.y1:.0f},{d.x2:.0f},{d.y2:.0f}) conf={d.confidence:.3f}")


if __name__ == "__main__":
    main()
