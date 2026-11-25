"""Turn saved dashboard annotations into YOLO datasets and train queued jobs."""

import asyncio
import hashlib
import json
import logging
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import asyncpg

from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)
WORKER_LOCK = 8700419


def split_photos(photos, part_name, options=None):
    """Hold out photos with diverse labels while keeping every class in training."""
    options = options or {}
    seed = options.get("seed")
    validation_percent = options.get("validation_percent", 20)
    test_percent = options.get("test_percent", 10)

    def order(photo):
        digest = photo["digest"]
        return digest if seed is None else hashlib.sha256(f"{seed}:{digest}".encode()).hexdigest()

    ordered = sorted(photos, key=order)
    positive = [photo for photo in ordered if photo["boxes"]]
    negative = [photo for photo in ordered if not photo["boxes"]]
    labels = {
        photo["id"]: {box.get("label") or part_name for box in photo["boxes"]}
        for photo in positive
    }
    remaining = Counter(label for names in labels.values() for label in names)
    def holdout(pool, counts, target):
        selected = []
        covered = set()
        for _ in range(target):
            candidates = [
                photo for photo in pool
                if all(counts[label] > 1 for label in labels[photo["id"]])
            ]
            if not candidates:
                break
            photo = max(candidates, key=lambda item: len(labels[item["id"]] - covered))
            names = labels[photo["id"]]
            covered.update(names)
            counts.subtract(names)
            selected.append(photo)
            pool.remove(photo)
        return selected, covered

    positive_count = len(positive)
    validation_target = max(1, round(positive_count * validation_percent / 100))
    test_target = max(1, round(positive_count * test_percent / 100))
    valid, _ = holdout(positive, remaining, validation_target)
    test = []
    if test_percent and len(photos) >= 5:
        test_candidates = positive.copy()
        test, covered = holdout(test_candidates, remaining.copy(), test_target)
        if covered == set(remaining):
            positive = test_candidates
        else:
            test = []
    negative_count = max(1, round(len(negative) * validation_percent / 100)) if len(negative) >= 2 else 0
    test_negative_count = min(
        len(negative) - negative_count - 1,
        max(1, round(len(negative) * test_percent / 100)),
    ) if test and len(negative) > negative_count + 1 else 0
    end = negative_count + test_negative_count
    splits = {"train": positive + negative[end:], "valid": valid + negative[:negative_count]}
    if test:
        splits["test"] = test + negative[negative_count:end]
    return splits


class TrainingInputError(Exception):
    pass


def make_dataset(root, image_root, job, snapshot):
    """Export named parts and YOLO's normalized center/size coordinates."""
    part_name = snapshot["part_name"]
    names = sorted({
        box.get("label") or part_name
        for photo in snapshot["photos"]
        for box in photo["boxes"]
    })
    class_ids = {name: index for index, name in enumerate(names)}
    splits = split_photos(snapshot["photos"], part_name, snapshot.get("options"))
    for split, photos in splits.items():
        images = root / split / "images"
        labels = root / split / "labels"
        images.mkdir(parents=True)
        labels.mkdir()
        for photo in photos:
            stem = photo["id"]
            suffix = "jpg" if photo["media_type"] == "image/jpeg" else "png"
            source = (
                image_root
                / snapshot["org_id"]
                / str(job["dataset_id"])
                / stem
                / f"{photo['digest']}.{suffix}"
            )
            try:
                with Image.open(source) as image:
                    image.load()
                shutil.copyfile(source, images / f"{stem}.{suffix}")
            except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
                name = photo.get("filename") or stem
                raise TrainingInputError(f"Cannot read {name}. Re-upload that photo and try again.") from exc
            lines = []
            for box in photo["boxes"]:
                a, b = box["start"], box["end"]
                x, y = (a["x"] + b["x"]) / 2, (a["y"] + b["y"]) / 2
                width, height = abs(a["x"] - b["x"]), abs(a["y"] - b["y"])
                class_id = class_ids[box.get("label") or part_name]
                lines.append(f"{class_id} {x:.9f} {y:.9f} {width:.9f} {height:.9f}\n")
            (labels / f"{stem}.txt").write_text("".join(lines))
    manifest = root / "data.yaml"
    manifest.write_text(
        json.dumps(
            {
                "path": ".",
                "train": "train/images",
                "val": "valid/images",
                "names": names,
                **({"test": "test/images"} if "test" in splits else {}),
            }
        )
    )
    return manifest


async def train(job, image_root, run_root, epochs, connection):
    snapshot = json.loads(job["snapshot"])
    options = snapshot.get("options") or {}
    pretrained = os.getenv("VI_TRAINING_WEIGHTS")
    if options.get("pretrained", True) and pretrained and not Path(pretrained).is_file():
        raise TrainingInputError("The Hugging Face starter weights are missing from the training worker.")
    output = run_root / snapshot["org_id"] / str(job["id"])
    output.mkdir(parents=True)
    weights_cache = run_root / "pretrained"
    weights_cache.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="vi-training-") as directory:
        manifest = await asyncio.to_thread(make_dataset, Path(directory), image_root, job, snapshot)
        if await training_cancelled(connection, job["id"]):
            return
        settings = Path(directory) / "training.json"
        settings.write_text(json.dumps({
            key: value for key, value in options.items() if key not in {
                "version_name", "notes", "max_minutes", "validation_percent", "test_percent", "min_box_px",
                "preset",
            }
        }))
        with (output / "training.log").open("wb") as log:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "vi_inference.train_detector",
                "--data",
                str(manifest),
                "--runs",
                str(output),
                "--epochs",
                str(epochs),
                "--options",
                str(settings),
                cwd=weights_cache,
                stdout=log,
                stderr=asyncio.subprocess.STDOUT,
            )
            try:
                async with asyncio.timeout(options.get("max_minutes", 120) * 60):
                    while process.returncode is None:
                        if await training_cancelled(connection, job["id"]):
                            return
                        try:
                            await asyncio.wait_for(process.wait(), timeout=2)
                        except TimeoutError:
                            continue
                    if process.returncode != 0:
                        raise RuntimeError("YOLO trainer exited unsuccessfully")
            finally:
                if process.returncode is None:
                    process.kill()
                    await process.wait()
        if await training_cancelled(connection, job["id"]):
            return
        result = output / "object-detection"
        checkpoint = result / "weights" / "best.pt"
        if not checkpoint.is_file() or not checkpoint.stat().st_size:
            raise RuntimeError("YOLO did not produce a checkpoint")
        checkpoint.replace(output / "best.pt")


async def training_cancelled(connection, job_id):
    status = await connection.fetchval("SELECT status FROM training_jobs WHERE id = $1", job_id)
    return status == "cancelled"


async def run():
    url = os.environ["DATABASE_URL"].replace("postgresql+asyncpg://", "postgresql://", 1)
    image_root = Path(os.environ["VI_TRAINING_IMAGE_DIR"]).resolve()
    run_root = Path(os.environ["VI_TRAINING_RUN_DIR"]).resolve()
    epochs = int(os.getenv("VI_TRAINING_EPOCHS", "20"))
    if epochs < 1:
        raise ValueError("VI_TRAINING_EPOCHS must be positive")
    connection = await asyncpg.connect(url)
    try:
        await connection.execute("SELECT pg_advisory_lock($1::bigint)", WORKER_LOCK)
        await connection.execute("""
            UPDATE training_jobs SET status = 'failed', finished_at = now(),
                error = 'The training worker restarted. Start training again.'
            WHERE status = 'running'
        """)
        while True:
            job = await connection.fetchrow("""
                UPDATE training_jobs SET status = 'running', started_at = now()
                WHERE id = (
                    SELECT id FROM training_jobs WHERE status = 'queued'
                    ORDER BY created_at, id LIMIT 1 FOR UPDATE SKIP LOCKED
                ) RETURNING id, dataset_id, snapshot
            """)
            if not job:
                await asyncio.sleep(2)
                continue
            status, error = "succeeded", None
            try:
                await train(job, image_root, run_root, epochs, connection)
            except TrainingInputError as exc:
                logger.warning("Invalid input for training job %s: %s", job["id"], exc)
                status, error = "failed", str(exc)
            except TimeoutError:
                logger.exception("Training job %s timed out", job["id"])
                status, error = "failed", "Training reached its time limit. Try a smaller photo set or a longer limit."
            except Exception:
                logger.exception("Training job %s failed", job["id"])
                status = "failed"
                error = "YOLO could not train on these photos. Check the images and try again."
            await connection.execute(
                """
                UPDATE training_jobs SET status = $2, error = $3, finished_at = now()
                WHERE id = $1 AND status = 'running'
            """,
                job["id"],
                status,
                error,
            )
    finally:
        await connection.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run())
