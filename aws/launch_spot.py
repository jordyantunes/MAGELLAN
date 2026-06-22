#!/usr/bin/env python3
"""
Launch an EC2 Spot Fleet for a MAGELLAN training run.

The fleet runs with Type=maintain so AWS automatically replaces a terminated
spot instance with a new one. The new instance resumes from the latest S3
checkpoint via user_data.tpl.sh.

Usage:
    python aws/launch_spot.py                          # uses spot_config.yaml
    python aws/launch_spot.py --config aws/spot_config.yaml
    python aws/launch_spot.py --sampler random --seed 1
    python aws/launch_spot.py --run-name run-002 --sampler magellan
    python aws/launch_spot.py --dry-run               # print request without launching
"""

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import boto3
import yaml
from botocore.exceptions import ClientError

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


def get_subnet_ids(cfg: dict) -> str | None:
    """Return subnet ID(s) for the launch spec.

    If subnet_id is set in config, use it as-is. Otherwise auto-discover all
    subnets in the default VPC and return them as a comma-separated list so
    Spot Fleet can pick whichever AZ has capacity.
    """
    subnet = cfg["aws"].get("subnet_id", "").strip()
    if subnet:
        return subnet
    ec2 = boto3.client("ec2", region_name=cfg["aws"]["region"])
    vpcs = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])
    if not vpcs["Vpcs"]:
        return None
    vpc_id = vpcs["Vpcs"][0]["VpcId"]
    subnets = ec2.describe_subnets(Filters=[{"Name": "vpc-id", "Values": [vpc_id]}])
    ids = [s["SubnetId"] for s in subnets["Subnets"]]
    if ids:
        print(f"Using subnets across {len(ids)} AZs: {', '.join(ids)}")
    return ",".join(ids) if ids else None


def get_fleet_role_arn(cfg: dict) -> str:
    """Return the IAM role ARN for Spot Fleet.

    Uses the explicit config value if set, otherwise looks for a role named
    AmazonEC2SpotFleetRole and creates it if it doesn't exist. This is a
    regular IAM role (not a service-linked role) with a trust policy for
    spotfleet.amazonaws.com, which is what RequestSpotFleet requires.
    """
    fleet_role = cfg["aws"].get("fleet_role_arn", "").strip()
    if fleet_role:
        return fleet_role

    iam = boto3.client("iam")
    role_name = "AmazonEC2SpotFleetRole"

    try:
        return iam.get_role(RoleName=role_name)["Role"]["Arn"]
    except iam.exceptions.NoSuchEntityException:
        pass

    print(f"IAM role '{role_name}' not found — creating it.")
    trust_policy = json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "spotfleet.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }],
    })
    resp = iam.create_role(
        RoleName=role_name,
        AssumeRolePolicyDocument=trust_policy,
        Description="Allows EC2 Spot Fleet to request and manage instances.",
    )
    iam.attach_role_policy(
        RoleName=role_name,
        PolicyArn="arn:aws:iam::aws:policy/service-role/AmazonEC2SpotFleetTaggingRole",
    )
    print(f"Created IAM role: {role_name}")
    return resp["Role"]["Arn"]


def create_spot_fleet(cfg: dict, user_data: str, dry_run: bool) -> str | None:
    ec2 = boto3.client("ec2", region_name=cfg["aws"]["region"])
    instance_cfg = cfg["instance"]
    aws_cfg = cfg["aws"]

    launch_spec: dict = {
        "ImageId": instance_cfg["ami_id"],
        "InstanceType": instance_cfg["type"],
        "KeyName": aws_cfg["key_pair"],
        "IamInstanceProfile": {"Name": aws_cfg["iam_instance_profile"]},
        "UserData": base64.b64encode(user_data.encode()).decode(),
        # Spot Fleet injects its own subnet ID at the instance level when
        # selecting an AZ, so NetworkInterfaces cannot be used here — use
        # top-level SecurityGroups and SubnetId instead.
        "SecurityGroups": [{"GroupId": aws_cfg["security_group_id"]}],
        **( {"SubnetId": subnet_ids} if (subnet_ids := get_subnet_ids(cfg)) else {} ),
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

    fleet_config: dict = {
        "AllocationStrategy": "lowestPrice",
        "TargetCapacity": 1,
        "Type": "maintain",
        "InstanceInterruptionBehavior": "terminate",
        "IamFleetRole": get_fleet_role_arn(cfg),
        "LaunchSpecifications": [launch_spec],
    }

    max_price = str(instance_cfg.get("spot_max_price") or "").strip()
    if max_price:
        fleet_config["SpotPrice"] = max_price

    if dry_run:
        printable = {**fleet_config}
        printable["LaunchSpecifications"] = [
            {k: ("<omitted>" if k == "UserData" else v) for k, v in launch_spec.items()}
        ]
        print("=== DRY RUN — would call ec2.request_spot_fleet with: ===")
        print(json.dumps(printable, indent=2))
        return None

    response = ec2.request_spot_fleet(SpotFleetRequestConfig=fleet_config)
    return response["SpotFleetRequestId"]


def wait_for_fleet_instance(ec2_client, fleet_id: str) -> tuple[str, str]:
    """Poll until the fleet has launched an instance; return (instance_id, public_ip)."""
    print(f"Waiting for fleet {fleet_id} to launch an instance", end="", flush=True)
    while True:
        resp = ec2_client.describe_spot_fleet_instances(SpotFleetRequestId=fleet_id)
        instances = resp.get("ActiveInstances", [])
        if instances:
            instance_id = instances[0]["InstanceId"]
            print(f"\nInstance launched: {instance_id}. Waiting for running state...",
                  end="", flush=True)
            waiter = ec2_client.get_waiter("instance_running")
            waiter.wait(InstanceIds=[instance_id])
            desc = ec2_client.describe_instances(InstanceIds=[instance_id])
            public_ip = desc["Reservations"][0]["Instances"][0].get("PublicIpAddress", "")
            print(f" running. Public IP: {public_ip}")
            return instance_id, public_ip
        print(".", end="", flush=True)
        time.sleep(5)


def check_required_fields(cfg: dict) -> list[str]:
    problems = []
    for placeholder in ("sg-XXXXXXXX", "ami-XXXXXXXX", "ACCOUNT_ID.dkr.ecr"):
        raw = str(cfg)
        if placeholder in raw:
            problems.append(placeholder)
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch MAGELLAN spot fleet")
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
    fleet_id = create_spot_fleet(cfg, user_data, args.dry_run)

    if fleet_id is None:
        return  # dry run

    print(f"Spot Fleet requested: {fleet_id}")

    ec2 = boto3.client("ec2", region_name=cfg["aws"]["region"])
    instance_id, public_ip = wait_for_fleet_instance(ec2, fleet_id)

    key_pair = cfg["aws"]["key_pair"]
    print()
    print("=== Connect ===")
    print(f"  ssh -i ~/.ssh/{key_pair}.pem ec2-user@{public_ip}")
    print()
    print("=== Monitor training log ===")
    print(f"  ssh -i ~/.ssh/{key_pair}.pem ec2-user@{public_ip} 'tail -f /var/log/magellan-init.log'")
    print()
    print("=== Stop training (cancels fleet so it does not relaunch) ===")
    print(f"  python aws/terminate_spot.py --fleet-id {fleet_id}")
    print()
    print(f"Fleet ID: {fleet_id}   Instance ID: {instance_id}")


if __name__ == "__main__":
    main()
