#!/usr/bin/env python3
"""
Launch an EC2 spot instance for a MAGELLAN training run.

Usage:
    python aws/launch_spot.py                          # uses spot_config.yaml
    python aws/launch_spot.py --config aws/spot_config.yaml
    python aws/launch_spot.py --sampler random --seed 1
    python aws/launch_spot.py --run-name run-002 --sampler magellan
    python aws/launch_spot.py --dry-run               # print request without launching
"""

import argparse
import base64
import sys
from pathlib import Path

import boto3
import yaml
from botocore.exceptions import ClientError

CONFIG_FILE = Path(__file__).parent / "spot_config.yaml"


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_user_data(cfg: dict, sampler: str, seed: int, run_name: str) -> str:
    template = (Path(__file__).parent / "user_data.sh").read_text()
    replacements = {
        "__BUCKET__": cfg["s3"]["bucket"],
        "__S3_PREFIX__": cfg["s3"]["prefix"],
        "__RUN_NAME__": run_name,
        "__DOCKER_IMAGE__": cfg["docker"]["image"],
        "__AWS_REGION__": cfg["aws"]["region"],
        "__SAMPLER__": sampler,
        "__SEED__": str(seed),
        "__SYNC_INTERVAL__": str(cfg["training"]["checkpoint_sync_interval"]),
    }
    for placeholder, value in replacements.items():
        template = template.replace(placeholder, value)
    return template


def request_spot_instance(cfg: dict, user_data: str, dry_run: bool) -> dict | None:
    ec2 = boto3.client("ec2", region_name=cfg["aws"]["region"])

    instance_cfg = cfg["instance"]
    aws_cfg = cfg["aws"]

    launch_spec = {
        "ImageId": instance_cfg["ami_id"],
        "InstanceType": instance_cfg["type"],
        "KeyName": aws_cfg["key_pair"],
        "IamInstanceProfile": {"Name": aws_cfg["iam_instance_profile"]},
        "UserData": base64.b64encode(user_data.encode()).decode(),
        # Network interface instead of top-level SecurityGroupIds so we can set
        # AssociatePublicIpAddress=True explicitly (required for SSH access).
        "NetworkInterfaces": [
            {
                "DeviceIndex": 0,
                "AssociatePublicIpAddress": True,
                "Groups": [aws_cfg["security_group_id"]],
                **( {"SubnetId": aws_cfg["subnet_id"].strip()}
                    if aws_cfg.get("subnet_id", "").strip() else {} ),
            }
        ],
        "BlockDeviceMappings": [
            {
                "DeviceName": "/dev/sda1",
                "Ebs": {
                    "VolumeSize": instance_cfg["volume_size_gb"],
                    "VolumeType": "gp3",
                    "DeleteOnTermination": True,
                },
            }
        ],
    }

    spot_options: dict = {
        "SpotInstanceType": "one-time",
        "InstanceInterruptionBehavior": "terminate",
    }
    max_price = str(instance_cfg.get("spot_max_price") or "").strip()
    if max_price:
        spot_options["MaxPrice"] = max_price

    run_kwargs = {
        "MinCount": 1,
        "MaxCount": 1,
        "InstanceMarketOptions": {
            "MarketType": "spot",
            "SpotOptions": spot_options,
        },
        "TagSpecifications": [
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": instance_cfg["instance_name"]},
                    {"Key": "Project", "Value": "MAGELLAN"},
                ],
            }
        ],
        **launch_spec,
    }

    if dry_run:
        print("=== DRY RUN — would call ec2.run_instances with: ===")
        import json
        printable = {k: v for k, v in run_kwargs.items() if k != "UserData"}
        printable["UserData"] = "<omitted>"
        print(json.dumps(printable, indent=2))
        return None

    response = ec2.run_instances(**run_kwargs)
    instance = response["Instances"][0]
    return instance


def wait_for_running(ec2_client, instance_id: str) -> str:
    print(f"Waiting for {instance_id} to reach running state...", end="", flush=True)
    waiter = ec2_client.get_waiter("instance_running")
    waiter.wait(InstanceIds=[instance_id])

    desc = ec2_client.describe_instances(InstanceIds=[instance_id])
    public_ip = desc["Reservations"][0]["Instances"][0].get("PublicIpAddress", "")
    print(f" running. Public IP: {public_ip}")
    return public_ip


def check_required_fields(cfg: dict) -> list[str]:
    problems = []
    for placeholder in ("sg-XXXXXXXX", "ami-XXXXXXXX", "ACCOUNT_ID.dkr.ecr"):
        raw = str(cfg)
        if placeholder in raw:
            problems.append(placeholder)
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch MAGELLAN spot instance")
    parser.add_argument("--config", type=Path, default=CONFIG_FILE)
    parser.add_argument("--sampler", help="Override training.sampler from config")
    parser.add_argument("--seed", type=int, help="Override training.seed from config")
    parser.add_argument("--run-name", help="Override training.run_name from config")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the launch request without actually launching")
    args = parser.parse_args()

    cfg = load_config(args.config)

    problems = check_required_fields(cfg)
    if problems and not args.dry_run:
        print("ERROR: spot_config.yaml still has unfilled placeholder values:")
        for p in problems:
            print(f"  {p}")
        print("Edit aws/spot_config.yaml and fill in the REQUIRED fields.")
        sys.exit(1)

    sampler = args.sampler or cfg["training"]["sampler"]
    seed = args.seed if args.seed is not None else cfg["training"]["seed"]
    run_name = args.run_name or cfg["training"]["run_name"]

    print(f"Sampler: {sampler}  seed: {seed}  run: {run_name}")
    print(f"Instance: {cfg['instance']['type']}  AMI: {cfg['instance']['ami_id']}")
    print(f"S3 checkpoints: s3://{cfg['s3']['bucket']}/{cfg['s3']['prefix']}/{run_name}/")

    user_data = build_user_data(cfg, sampler, seed, run_name)
    instance = request_spot_instance(cfg, user_data, args.dry_run)

    if instance is None:
        return  # dry run

    instance_id = instance["InstanceId"]
    print(f"Instance requested: {instance_id}")

    ec2 = boto3.client("ec2", region_name=cfg["aws"]["region"])
    public_ip = wait_for_running(ec2, instance_id)

    key_pair = cfg["aws"]["key_pair"]
    print()
    print("=== Connect ===")
    print(f"  ssh -i ~/.ssh/{key_pair}.pem ec2-user@{public_ip}")
    print()
    print("=== Monitor training log ===")
    print(f"  ssh -i ~/.ssh/{key_pair}.pem ec2-user@{public_ip} 'tail -f /var/log/magellan-init.log'")
    print()
    print("=== Terminate when done ===")
    print(f"  python aws/terminate_spot.py --instance-id {instance_id}")
    print()
    print(f"Instance ID saved for terminate_spot.py: {instance_id}")


if __name__ == "__main__":
    main()
