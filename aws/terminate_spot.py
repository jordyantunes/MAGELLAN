#!/usr/bin/env python3
"""
Cancel a MAGELLAN Spot Fleet and optionally trigger a final S3 sync first.

Cancelling the fleet (not just terminating the instance) prevents AWS from
automatically relaunching a replacement. Use --fleet-id when you have it;
otherwise the script looks it up from the instance's AWS-managed tag.

Usage:
    python aws/terminate_spot.py --fleet-id sfr-...
    python aws/terminate_spot.py --instance-id i-0abc123
    python aws/terminate_spot.py --name magellan-training
    python aws/terminate_spot.py --fleet-id sfr-... --no-sync
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


def get_fleet_id_for_instance(ec2_client, instance_id: str) -> str | None:
    """Return the Spot Fleet request ID for a fleet-managed instance, or None."""
    resp = ec2_client.describe_instances(InstanceIds=[instance_id])
    tags = resp["Reservations"][0]["Instances"][0].get("Tags", [])
    for tag in tags:
        if tag["Key"] == "aws:ec2spot:fleet-request-id":
            return tag["Value"]
    return None


def get_instance_for_fleet(ec2_client, fleet_id: str) -> str | None:
    resp = ec2_client.describe_spot_fleet_instances(SpotFleetRequestId=fleet_id)
    instances = resp.get("ActiveInstances", [])
    return instances[0]["InstanceId"] if instances else None


def trigger_remote_sync(public_ip: str, key_pair: str, run_name: str,
                        bucket: str, prefix: str) -> None:
    import subprocess
    s3_path = f"s3://{bucket}/{prefix}/{run_name}/"
    local_path = f"/home/ec2-user/outputs/{run_name}/"
    cmd = f"aws s3 sync {local_path} {s3_path}"
    ssh_cmd = [
        "ssh", "-o", "StrictHostKeyChecking=no",
        "-i", f"~/.ssh/{key_pair}.pem",
        f"ec2-user@{public_ip}",
        cmd,
    ]
    print(f"Running remote sync: {cmd}")
    result = subprocess.run(ssh_cmd, timeout=120)
    if result.returncode != 0:
        print("Warning: remote sync returned non-zero — instance may have already synced or be unreachable.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cancel MAGELLAN Spot Fleet")
    parser.add_argument("--config", type=Path, default=CONFIG_FILE)
    parser.add_argument("--fleet-id", help="Spot Fleet request ID (sfr-...)")
    parser.add_argument("--instance-id", help="EC2 instance ID — fleet ID looked up from tags")
    parser.add_argument("--name", help="Find instance by Name tag (default: instance_name from config)")
    parser.add_argument("--no-sync", action="store_true",
                        help="Skip the final S3 sync before cancellation")
    parser.add_argument("--run-name", help="Override training.run_name from config (for sync path)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    region = cfg["aws"]["region"]
    ec2 = boto3.client("ec2", region_name=region)

    fleet_id = args.fleet_id
    instance_id = args.instance_id

    # Resolve instance ID from name tag if needed
    if not fleet_id and not instance_id:
        search_name = args.name or cfg["instance"]["instance_name"]
        ids = find_instance_by_name(ec2, search_name)
        if not ids:
            print(f"No running instances found with name '{search_name}'")
            sys.exit(1)
        if len(ids) > 1:
            print(f"Multiple instances found: {ids}")
            print("Specify --fleet-id or --instance-id to be explicit.")
            sys.exit(1)
        instance_id = ids[0]
        print(f"Found instance: {instance_id}")

    # Resolve fleet ID from instance tags if we only have an instance ID
    if not fleet_id and instance_id:
        fleet_id = get_fleet_id_for_instance(ec2, instance_id)
        if fleet_id:
            print(f"Fleet ID from instance tags: {fleet_id}")
        else:
            print("Warning: instance does not appear to be fleet-managed. Will terminate instance directly.")

    # Resolve instance ID from fleet if we only have a fleet ID
    if fleet_id and not instance_id:
        instance_id = get_instance_for_fleet(ec2, fleet_id)

    # Optional sync before shutdown
    if not args.no_sync and instance_id:
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

    # Cancel fleet (terminates the instance and prevents relaunch) or fall back
    # to direct instance termination for non-fleet instances.
    if fleet_id:
        print(f"Cancelling fleet {fleet_id} (TerminateInstances=True)...")
        ec2.cancel_spot_fleet_requests(
            SpotFleetRequestIds=[fleet_id],
            TerminateInstances=True,
        )
        print("Fleet cancelled. The instance will shut down shortly.")
    elif instance_id:
        print(f"Terminating instance {instance_id} directly...")
        ec2.terminate_instances(InstanceIds=[instance_id])
        print("Termination requested.")

    print(f"Checkpoints are at: s3://{cfg['s3']['bucket']}/{cfg['s3']['prefix']}/{cfg['training']['run_name']}/")


if __name__ == "__main__":
    main()
