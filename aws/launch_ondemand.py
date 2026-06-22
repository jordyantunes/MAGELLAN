#!/usr/bin/env python3
"""
Launch an on-demand EC2 instance for a MAGELLAN training run.

Uses the same user_data.tpl.sh and spot_config.yaml as the spot fleet script.
Terminate with: python aws/terminate_spot.py --instance-id <id>

Usage:
    python aws/launch_ondemand.py
    python aws/launch_ondemand.py --sampler random --seed 1
    python aws/launch_ondemand.py --run-name run-002 --sampler magellan
    python aws/launch_ondemand.py --dry-run
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import boto3
import yaml

CONFIG_FILE = Path(__file__).parent / "spot_config.yaml"


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_user_data(cfg: dict, sampler: str, seed: int, run_name: str) -> str:
    template = (Path(__file__).parent / "user_data.tpl.sh").read_text()
    compose_content = (Path(__file__).parent.parent / "docker-compose.aws.yml").read_text()
    replacements = {
        "__BUCKET__": cfg["s3"]["bucket"],
        "__S3_PREFIX__": cfg["s3"]["prefix"],
        "__RUN_NAME__": run_name,
        "__DOCKER_IMAGE__": cfg["docker"]["image"],
        "__AWS_REGION__": cfg["aws"]["region"],
        "__SAMPLER__": sampler,
        "__SEED__": str(seed),
        "__SYNC_INTERVAL__": str(cfg["training"]["checkpoint_sync_interval"]),
        "__DOCKER_COMPOSE_CONTENT__": compose_content,
    }
    for placeholder, value in replacements.items():
        template = template.replace(placeholder, value)
    return template


def launch_ondemand(cfg: dict, user_data: str, dry_run: bool) -> str | None:
    ec2 = boto3.client("ec2", region_name=cfg["aws"]["region"])
    instance_cfg = cfg["instance"]
    aws_cfg = cfg["aws"]

    run_kwargs = {
        "MinCount": 1,
        "MaxCount": 1,
        "ImageId": instance_cfg["ami_id"],
        "InstanceType": instance_cfg["type"],
        "KeyName": aws_cfg["key_pair"],
        "IamInstanceProfile": {"Name": aws_cfg["iam_instance_profile"]},
        "UserData": base64.b64encode(user_data.encode()).decode(),
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
                "DeviceName": "/dev/xvda",
                "Ebs": {
                    "VolumeSize": instance_cfg["volume_size_gb"],
                    "VolumeType": "gp3",
                    "DeleteOnTermination": True,
                },
            }
        ],
        "TagSpecifications": [
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": instance_cfg["instance_name"]},
                    {"Key": "Project", "Value": "MAGELLAN"},
                ],
            }
        ],
    }

    if dry_run:
        printable = {k: ("<omitted>" if k == "UserData" else v) for k, v in run_kwargs.items()}
        print("=== DRY RUN — would call ec2.run_instances with: ===")
        print(json.dumps(printable, indent=2))
        return None

    response = ec2.run_instances(**run_kwargs)
    return response["Instances"][0]["InstanceId"]


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
        if placeholder in str(cfg):
            problems.append(placeholder)
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch MAGELLAN on-demand instance")
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
        sys.exit(1)

    sampler = args.sampler or cfg["training"]["sampler"]
    seed = args.seed if args.seed is not None else cfg["training"]["seed"]
    run_name = args.run_name or cfg["training"]["run_name"]

    print(f"Sampler: {sampler}  seed: {seed}  run: {run_name}")
    print(f"Instance: {cfg['instance']['type']}  AMI: {cfg['instance']['ami_id']}")
    print(f"S3 checkpoints: s3://{cfg['s3']['bucket']}/{cfg['s3']['prefix']}/{run_name}/")

    user_data = build_user_data(cfg, sampler, seed, run_name)
    instance_id = launch_ondemand(cfg, user_data, args.dry_run)

    if instance_id is None:
        return  # dry run

    print(f"Instance launched: {instance_id}")

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


if __name__ == "__main__":
    main()
