#!/usr/bin/env python3
"""
Create the IAM role, instance profile, and security group required to run
MAGELLAN on EC2 Spot.

The role (magellan-ec2-role) grants the EC2 instance permission to:
  - Pull the Docker image from ECR
  - Read/write checkpoints to the S3 bucket

The security group allows inbound SSH (port 22) from anywhere so you can
monitor the instance. It is created automatically if aws.security_group_id is
not set in spot_config.yaml, and the ID is written back to the file.

Run once before your first launch:
    python aws/setup_iam.py
    python aws/setup_iam.py --dry-run   # print what would be created

All names and the bucket are read from aws/spot_config.yaml so they stay in
sync with launch_spot.py automatically.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import boto3
import yaml
from botocore.exceptions import ClientError

CONFIG_FILE = Path(__file__).parent / "spot_config.yaml"

TRUST_POLICY = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"Service": "ec2.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }
    ],
}


def s3_policy(bucket: str) -> dict:
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "CheckpointAccess",
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:DeleteObject",
                    "s3:ListBucket",
                ],
                "Resource": [
                    f"arn:aws:s3:::{bucket}",
                    f"arn:aws:s3:::{bucket}/*",
                ],
            }
        ],
    }


ECR_POLICY = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "ECRAuth",
            "Effect": "Allow",
            "Action": "ecr:GetAuthorizationToken",
            "Resource": "*",
        },
        {
            "Sid": "ECRPull",
            "Effect": "Allow",
            "Action": [
                "ecr:BatchGetImage",
                "ecr:GetDownloadUrlForLayer",
                "ecr:BatchCheckLayerAvailability",
            ],
            "Resource": "*",
        },
    ],
}


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_config_value(path: Path, key_path: str, value: str) -> None:
    """Write a single scalar value back into the YAML file, preserving all comments."""
    text = path.read_text()
    # key_path is dot-separated, e.g. "aws.security_group_id" → last segment is the key
    key = key_path.split(".")[-1]
    # Replace the value on the line that starts with the key (with optional leading spaces)
    text = re.sub(
        rf"^(\s*{re.escape(key)}:\s*).*$",
        rf"\g<1>{value}",
        text,
        flags=re.MULTILINE,
    )
    path.write_text(text)


def get_default_vpc(ec2, region: str) -> str:
    resp = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])
    vpcs = resp.get("Vpcs", [])
    if not vpcs:
        print(f"No default VPC found in {region}. Create one or set aws.subnet_id manually.")
        sys.exit(1)
    return vpcs[0]["VpcId"]


def ensure_security_group(ec2, cfg: dict, config_path: Path, dry_run: bool) -> str:
    sg_id = cfg["aws"].get("security_group_id", "").strip()
    if sg_id and sg_id != "sg-XXXXXXXX":
        print(f"Security group already set: {sg_id}")
        return sg_id

    region = cfg["aws"]["region"]
    sg_name = "magellan-sg"

    if dry_run:
        print(f"[dry-run] Would create security group: {sg_name}")
        print(f"  Inbound: TCP 22 (SSH) from 0.0.0.0/0")
        print(f"  Inbound: TCP 5000 (MLflow UI) from 0.0.0.0/0")
        print(f"  Would write sg-XXXXXXXX → spot_config.yaml aws.security_group_id")
        return "sg-XXXXXXXX"

    # Reuse existing group with the same name if it exists
    existing = ec2.describe_security_groups(
        Filters=[{"Name": "group-name", "Values": [sg_name]}]
    )["SecurityGroups"]
    if existing:
        sg_id = existing[0]["GroupId"]
        print(f"Security group '{sg_name}' already exists: {sg_id}")
    else:
        vpc_id = cfg["aws"].get("subnet_id", "").strip() or get_default_vpc(ec2, region)
        # If subnet_id was given, resolve its VPC
        if cfg["aws"].get("subnet_id", "").strip():
            subnet = ec2.describe_subnets(
                SubnetIds=[cfg["aws"]["subnet_id"].strip()]
            )["Subnets"][0]
            vpc_id = subnet["VpcId"]

        resp = ec2.create_security_group(
            GroupName=sg_name,
            Description="MAGELLAN EC2 Spot - SSH and MLflow access",
            VpcId=vpc_id,
        )
        sg_id = resp["GroupId"]

        ec2.authorize_security_group_ingress(
            GroupId=sg_id,
            IpPermissions=[
                {
                    "IpProtocol": "tcp",
                    "FromPort": 22,
                    "ToPort": 22,
                    "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "SSH"}],
                },
                {
                    "IpProtocol": "tcp",
                    "FromPort": 5000,
                    "ToPort": 5000,
                    "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "MLflow UI"}],
                },
            ],
        )
        print(f"Created security group: {sg_name} ({sg_id}) in VPC {vpc_id}")
        print(f"  Inbound rule: TCP 22 (SSH) from 0.0.0.0/0")
        print(f"  Inbound rule: TCP 5000 (MLflow) from 0.0.0.0/0")

    save_config_value(config_path, "aws.security_group_id", sg_id)
    print(f"  Written to spot_config.yaml: aws.security_group_id = {sg_id}")
    return sg_id


def role_exists(iam, role_name: str) -> bool:
    try:
        iam.get_role(RoleName=role_name)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchEntity":
            return False
        raise


def instance_profile_exists(iam, profile_name: str) -> bool:
    try:
        iam.get_instance_profile(InstanceProfileName=profile_name)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchEntity":
            return False
        raise


def create_or_update_role(iam, role_name: str, bucket: str, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] Would create IAM role: {role_name}")
        print(f"  Trust policy: EC2 service")
        print(f"  Inline policy 'S3CheckpointAccess': s3:::{bucket}/*")
        print(f"  Inline policy 'ECRPullAccess': ecr:* (all repos)")
        return

    if role_exists(iam, role_name):
        print(f"Role '{role_name}' already exists — updating inline policies.")
    else:
        iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(TRUST_POLICY),
            Description="Allows MAGELLAN EC2 instances to access S3 checkpoints and ECR.",
        )
        print(f"Created role: {role_name}")

    iam.put_role_policy(
        RoleName=role_name,
        PolicyName="S3CheckpointAccess",
        PolicyDocument=json.dumps(s3_policy(bucket)),
    )
    print(f"  Attached inline policy: S3CheckpointAccess (bucket: {bucket})")

    iam.put_role_policy(
        RoleName=role_name,
        PolicyName="ECRPullAccess",
        PolicyDocument=json.dumps(ECR_POLICY),
    )
    print(f"  Attached inline policy: ECRPullAccess")


def create_instance_profile(iam, role_name: str, dry_run: bool) -> None:
    profile_name = role_name  # keep profile name == role name (matches spot_config.yaml)

    if dry_run:
        print(f"[dry-run] Would create instance profile: {profile_name}")
        print(f"  Would add role '{role_name}' to profile")
        return

    if instance_profile_exists(iam, profile_name):
        print(f"Instance profile '{profile_name}' already exists — skipping creation.")
        # Still make sure the role is associated
        profile = iam.get_instance_profile(InstanceProfileName=profile_name)
        existing_roles = [r["RoleName"] for r in profile["InstanceProfile"]["Roles"]]
        if role_name not in existing_roles:
            iam.add_role_to_instance_profile(
                InstanceProfileName=profile_name,
                RoleName=role_name,
            )
            print(f"  Added role '{role_name}' to existing profile.")
    else:
        iam.create_instance_profile(InstanceProfileName=profile_name)
        iam.add_role_to_instance_profile(
            InstanceProfileName=profile_name,
            RoleName=role_name,
        )
        print(f"Created instance profile: {profile_name}")
        print(f"  Added role '{role_name}' to profile.")


def ensure_s3_bucket(bucket: str, region: str, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] Would ensure S3 bucket exists: {bucket} (region: {region})")
        return

    s3 = boto3.client("s3", region_name=region)
    try:
        s3.head_bucket(Bucket=bucket)
        print(f"S3 bucket already exists: {bucket}")
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("404", "NoSuchBucket"):
            kwargs = {"Bucket": bucket}
            if region != "us-east-1":
                kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region}
            s3.create_bucket(**kwargs)
            # Block all public access
            s3.put_public_access_block(
                Bucket=bucket,
                PublicAccessBlockConfiguration={
                    "BlockPublicAcls": True,
                    "IgnorePublicAcls": True,
                    "BlockPublicPolicy": True,
                    "RestrictPublicBuckets": True,
                },
            )
            print(f"Created S3 bucket: {bucket} (private, region: {region})")
        elif code == "403":
            print(f"S3 bucket '{bucket}' exists but is owned by another account — check your config.")
            sys.exit(1)
        else:
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up IAM role for MAGELLAN EC2 Spot")
    parser.add_argument("--config", type=Path, default=CONFIG_FILE)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be created without making any changes")
    args = parser.parse_args()

    cfg = load_config(args.config)
    region = cfg["aws"]["region"]
    role_name = cfg["aws"]["iam_instance_profile"]
    bucket = cfg["s3"]["bucket"]

    print(f"Region:           {region}")
    print(f"Role/profile:     {role_name}")
    print(f"S3 bucket:        {bucket}")
    print()

    iam = boto3.client("iam", region_name=region)
    ec2 = boto3.client("ec2", region_name=region)

    ensure_s3_bucket(bucket, region, args.dry_run)
    create_or_update_role(iam, role_name, bucket, args.dry_run)
    create_instance_profile(iam, role_name, args.dry_run)
    ensure_security_group(ec2, cfg, args.config, args.dry_run)

    if not args.dry_run:
        print()
        print("Done. All prerequisites are ready.")
        print(f"Remaining REQUIRED fields in aws/spot_config.yaml:")
        print(f"  - instance.ami_id  (run the aws ec2 describe-images command in the comment)")
        print(f"  - docker.image     (your ECR URI)")
        print(f"Then: python aws/launch_spot.py")


if __name__ == "__main__":
    main()
