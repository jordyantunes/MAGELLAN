#!/usr/bin/env python3
"""
Terminate a MAGELLAN spot instance and optionally do a final S3 sync first.

Usage:
    python aws/terminate_spot.py --instance-id i-0abc123
    python aws/terminate_spot.py --instance-id i-0abc123 --no-sync
    python aws/terminate_spot.py --name magellan-training   # find by Name tag
"""

import argparse
import sys
from pathlib import Path

import boto3
import yaml

CONFIG_FILE = Path(__file__).parent / "spot_config.yaml"


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def find_instance_by_name(ec2_client, name: str) -> list[str]:
    resp = ec2_client.describe_instances(
        Filters=[
            {"Name": "tag:Name", "Values": [name]},
            {"Name": "instance-state-name", "Values": ["pending", "running", "stopping"]},
        ]
    )
    return [
        i["InstanceId"]
        for r in resp["Reservations"]
        for i in r["Instances"]
    ]


def get_public_ip(ec2_client, instance_id: str) -> str:
    resp = ec2_client.describe_instances(InstanceIds=[instance_id])
    return resp["Reservations"][0]["Instances"][0].get("PublicIpAddress", "")


def trigger_remote_sync(public_ip: str, key_pair: str, run_name: str,
                        bucket: str, prefix: str) -> None:
    import subprocess
    s3_path = f"s3://{bucket}/{prefix}/{run_name}/"
    local_path = f"/home/ubuntu/outputs/{run_name}/"
    cmd = (
        f"aws s3 sync {local_path} {s3_path}"
    )
    ssh_cmd = [
        "ssh", "-o", "StrictHostKeyChecking=no",
        "-i", f"~/.ssh/{key_pair}.pem",
        f"ubuntu@{public_ip}",
        cmd,
    ]
    print(f"Running remote sync: {cmd}")
    result = subprocess.run(ssh_cmd, timeout=120)
    if result.returncode != 0:
        print("Warning: remote sync returned non-zero — instance may have already synced or be unreachable.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Terminate MAGELLAN spot instance")
    parser.add_argument("--config", type=Path, default=CONFIG_FILE)
    parser.add_argument("--instance-id", help="EC2 instance ID (i-...)")
    parser.add_argument("--name", help="Find instance by Name tag (default: instance_name from config)")
    parser.add_argument("--no-sync", action="store_true",
                        help="Skip the final S3 sync before termination")
    parser.add_argument("--run-name", help="Override training.run_name from config (for sync path)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    region = cfg["aws"]["region"]
    ec2 = boto3.client("ec2", region_name=region)

    if args.instance_id:
        instance_ids = [args.instance_id]
    else:
        search_name = args.name or cfg["instance"]["instance_name"]
        instance_ids = find_instance_by_name(ec2, search_name)
        if not instance_ids:
            print(f"No running instances found with name '{search_name}'")
            sys.exit(1)
        print(f"Found instances: {instance_ids}")
        if len(instance_ids) > 1:
            print("Multiple instances found — specify --instance-id to be explicit.")
            sys.exit(1)

    instance_id = instance_ids[0]

    if not args.no_sync:
        run_name = args.run_name or cfg["training"]["run_name"]
        public_ip = get_public_ip(ec2, instance_id)
        if public_ip:
            trigger_remote_sync(
                public_ip=public_ip,
                key_pair=cfg["aws"]["key_pair"],
                run_name=run_name,
                bucket=cfg["s3"]["bucket"],
                prefix=cfg["s3"]["prefix"],
            )
        else:
            print("Warning: no public IP found — skipping remote sync.")

    print(f"Terminating {instance_id}...")
    ec2.terminate_instances(InstanceIds=[instance_id])
    print("Termination requested. The instance will shut down shortly.")
    print(f"Checkpoints are at: s3://{cfg['s3']['bucket']}/{cfg['s3']['prefix']}/{cfg['training']['run_name']}/")


if __name__ == "__main__":
    main()
