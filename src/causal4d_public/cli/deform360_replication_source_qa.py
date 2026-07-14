from __future__ import annotations

import argparse

from causal4d_public.deform360_object_sam2 import (
    DeformableObjectSam2MaskConfig,
    DeformableObjectSam2VideoPredictor,
)
from causal4d_public.deform360_replication import (
    load_deform360_replication_protocol,
)
from causal4d_public.deform360_replication_source_qa import (
    load_source_qa_policy,
    run_source_geometry_qa,
    write_source_qa_artifact,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the locked source-only Deform360 multiview QA."
    )
    parser.add_argument("--replication-config", required=True)
    parser.add_argument("--source-qa-config", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--sam2-repository", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    replication = load_deform360_replication_protocol(args.replication_config)
    policy = load_source_qa_policy(args.source_qa_config)
    predictor = DeformableObjectSam2VideoPredictor(
        args.sam2_repository,
        args.checkpoint,
        device=args.device,
        config=DeformableObjectSam2MaskConfig(**policy["config"]["sam2"]),
    )
    try:
        artifact = run_source_geometry_qa(
            args.raw_root,
            replication,
            policy,
            predictor,
            args.output_dir,
        )
    finally:
        predictor.close()
    write_source_qa_artifact(args.artifact, artifact)
    print(artifact["result_sha256"])


if __name__ == "__main__":
    main()
